import numpy as np
from scipy.spatial.transform import Rotation as R

import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_


class BaseModel(nn.Module):
    
    @property
    def num_parameters(self):
        nweights, nparams = 0, 0
        for k, v in self.named_parameters():
            nweights += np.prod(v.size())
            nparams += 1
        return nweights, nparams

    @property
    def num_trainable_parameters(self):
        nweights, nparams = 0, 0
        for k, v in self.named_parameters():
            if v.requires_grad:
                nweights += np.prod(v.size())
                nparams += 1
        return nweights, nparams

    def prepare_batch(self, batch):
        device = next(self.parameters()).device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        return batch
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            trunc_normal_(m.weight, std=.02)


class RobotPoseEmbedding(nn.Module):
    def __init__(self, hidden_size) -> None:
        super().__init__()

        self.open_embedding = nn.Embedding(2, hidden_size)
        self.pos_embedding = nn.Linear(3, hidden_size)
        self.rot_embedding = nn.Linear(6, hidden_size)  # sin and cos of the euler angles
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, actions):
        '''
        actions: (batch_size, 8)
        '''
        pos_embeds = self.pos_embedding(actions[..., :3])
        open_embeds = self.open_embedding(actions[..., -1].long())

        rot_euler_angles = R.from_quat(actions[..., 3:7].data.cpu()).as_euler('xyz')
        rot_euler_angles = torch.from_numpy(rot_euler_angles).float().to(actions.device)
        rot_inputs = torch.cat(
            [torch.sin(rot_euler_angles), torch.cos(rot_euler_angles)], -1
        )
        rot_embeds = self.rot_embedding(rot_inputs)

        act_embeds = self.layer_norm(
            pos_embeds + rot_embeds + open_embeds
        )
        return act_embeds

