import torch
import torch.nn as nn

try:
    import flash_attn
except:
    print('No flash attn')

from genrobo3d.train.utils.ops import pad_tensors_wgrad, gen_seq_masks

from functools import partial

from genrobo3d.models.PointTransformerV3.model import (
    Point, PointModule, PointSequential, MLP, DropPath, PDNorm,
    Embedding, Block, SerializedPooling, SerializedUnpooling,
    PointTransformerV3, offset2bincount
)

class CrossAttention(PointModule):
    def __init__(
        self, channels, num_heads, kv_channels=None, attn_drop=0, proj_drop=0, 
        qk_norm=False, enable_flash=True
    ):
        super().__init__()
        if kv_channels is None:
            kv_channels = channels
        self.q = nn.Linear(channels, channels, bias=True)
        self.kv = nn.Linear(kv_channels, channels * 2, bias=True)
        self.attn_drop = attn_drop

        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.qk_norm = qk_norm
        self.enable_flash = enable_flash

        # TODO: eps should be 1 / 65530 if using fp16 (eps=1e-6)
        self.q_norm = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1e-6) if self.qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1e-6) if self.qk_norm else nn.Identity()

    def forward(self, point: Point):
        device = point.feat.device

        q = self.q(point.feat).view(-1, self.num_heads, self.head_dim)
        kv = self.kv(point.context).view(-1, 2, self.num_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(kv[:, 0])
        kv = torch.stack([k, kv[:, 1]], dim=1)

        if self.enable_flash:
            cu_seqlens_q = torch.cat([torch.zeros(1).int().to(device), point.offset.int()], dim=0)
            cu_seqlens_k = torch.cat([torch.zeros(1).int().to(device), point.context_offset.int()], dim=0)
            max_seqlen_q = offset2bincount(point.offset).max()
            max_seqlen_k = offset2bincount(point.context_offset).max()

            feat = flash_attn.flash_attn_varlen_kvpacked_func(
                q.half(), kv.half(), cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale
            ).reshape(-1, self.channels)
            feat = feat.to(q.dtype)
        else:
            # q: (#all points, #heads, #dim)
            # kv: (#all words, k/v, #heads, #dim)
            # print(q.size(), kv.size())
            npoints_in_batch = offset2bincount(point.offset).data.cpu().numpy().tolist()
            nwords_in_batch = offset2bincount(point.context_offset).data.cpu().numpy().tolist()
            word_padded_masks = torch.from_numpy(
                gen_seq_masks(nwords_in_batch)
            ).to(q.device).logical_not()
            # print(word_padded_masks)

            q_pad = pad_tensors_wgrad(
                torch.split(q, npoints_in_batch, dim=0), npoints_in_batch
            )
            kv_pad = pad_tensors_wgrad(
                torch.split(kv, nwords_in_batch), nwords_in_batch
            )
            # q_pad: (batch_size, #points, #heads, #dim)
            # kv_pad: (batch_size, #words, k/v, #heads, #dim)
            # print(q_pad.size(), kv_pad.size())
            logits = torch.einsum('bphd,bwhd->bpwh', q_pad, kv_pad[:, :, 0]) * self.scale
            logits.masked_fill_(word_padded_masks.unsqueeze(1).unsqueeze(-1), -1e4)
            attn_probs = torch.softmax(logits, dim=2)
            # print(attn_probs.size())
            feat = torch.einsum('bpwh,bwhd->bphd', attn_probs, kv_pad[:, :, 1])
            feat = torch.cat([ft[:npoints_in_batch[i]] for i, ft in enumerate(feat)], 0)
            feat = feat.reshape(-1, self.channels).float()
            # print(feat.size())

        # ffn
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point
    

class CABlock(PointModule):
    def __init__(
        self, channels, num_heads, kv_channels=None, attn_drop=0.0, proj_drop=0.0,
        mlp_ratio=4.0, norm_layer=nn.LayerNorm, act_layer=nn.GELU, pre_norm=True,
        qk_norm=False, enable_flash=True,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = CrossAttention(
            channels=channels,
            num_heads=num_heads,
            kv_channels=kv_channels,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            qk_norm=qk_norm,
            enable_flash=enable_flash,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )

    def forward(self, point: Point):
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point = self.attn(point)
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.mlp(point)
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class PointTransformerV3CA(PointTransformerV3):
    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        ctx_channels=256,
        qkv_bias=True,
        qk_scale=None,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_context_channels=256,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
        pdnorm_only_decoder=False,
        add_coords_in_attn=False,
        scaled_cosine_attn=False, # TODO
    ):
        PointModule.__init__(self)
        # assert enable_flash, 'only implemented flash attention'

        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.cls_mode or self.num_stages == len(dec_depths) + 1
        assert self.cls_mode or self.num_stages == len(dec_channels) + 1
        assert self.cls_mode or self.num_stages == len(dec_num_head) + 1
        assert self.cls_mode or self.num_stages == len(dec_patch_size) + 1

        # norm layers
        if pdnorm_bn:
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(
                    nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
                ),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
                context_channels=pdnorm_context_channels,
            )
        else:
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        if pdnorm_ln:
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
                context_channels=pdnorm_context_channels,
            )
        else:
            ln_layer = nn.LayerNorm
        # activation layers
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        # encoder
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                        shuffle_orders=self.shuffle_orders,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                        add_coords_in_attn=add_coords_in_attn,
                        qk_norm=qk_norm,
                    ),
                    name=f"block{i}",
                )
                if (not pdnorm_only_decoder) or (s == self.num_stages - 1):
                    enc.add(
                        CABlock(
                            channels=enc_channels[s],
                            num_heads=enc_num_head[s],
                            kv_channels=ctx_channels,
                            mlp_ratio=mlp_ratio,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            qk_norm=qk_norm,
                            enable_flash=enable_flash,
                        ),
                        name=f"ca_block{i}",
                    )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # decoder
        if not self.cls_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    SerializedUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                            add_coords_in_attn=add_coords_in_attn,
                            qk_norm=qk_norm,
                        ),
                        name=f"block{i}",
                    )
                    dec.add(
                        CABlock(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            kv_channels=ctx_channels,
                            mlp_ratio=mlp_ratio,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            qk_norm=qk_norm,
                            enable_flash=enable_flash,
                        ),
                        name=f"ca_block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")

    def forward(self, data_dict, return_dec_layers=False):
        """
        A data_dict is a dictionary containing properties of a batched point cloud.
        It should contain the following properties for PTv3:
        1. "feat": feature of point cloud
        2. "grid_coord": discrete coordinate after grid sampling (voxelization) or "coord" + "grid_size"
        3. "offset" or "batch": https://github.com/Pointcept/Pointcept?tab=readme-ov-file#offset
        """
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()
        # print('before', offset2bincount(point.offset))

        point = self.embedding(point)
        point = self.enc(point)
        # print('after', offset2bincount(point.offset))

        layer_outputs = [self._pack_point_dict(point)]

        if not self.cls_mode:
            if return_dec_layers:
                for i in range(len(self.dec)):
                    for dec_block in self.dec[i]:
                        point = dec_block(point)
                        if type(dec_block) == CABlock:
                            layer_outputs.append(self._pack_point_dict(point))
                return layer_outputs
            else:
                point = self.dec(point)
        return point