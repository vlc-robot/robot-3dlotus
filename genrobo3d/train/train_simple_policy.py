import os
import sys
import json
import argparse
import time
from collections import defaultdict
from tqdm import tqdm
import copy
from functools import partial

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

from genrobo3d.train.utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from genrobo3d.train.utils.save import ModelSaver, save_training_meta
from genrobo3d.train.utils.misc import NoOp, set_dropout, set_random_seed
from genrobo3d.train.utils.distributed import set_cuda, wrap_model, all_gather

from genrobo3d.train.optim import get_lr_sched, get_lr_sched_decay_rate
from genrobo3d.train.optim.misc import build_optimizer

from genrobo3d.configs.default import get_config

from genrobo3d.train.datasets.loader import build_dataloader
from genrobo3d.train.datasets.simple_policy_dataset import (
    SimplePolicyDataset, base_collate_fn, ptv3_collate_fn
)

from genrobo3d.models.simple_policy_ptv3 import (
    SimplePolicyPTV3AdaNorm, SimplePolicyPTV3CA, SimplePolicyPTV3Concat
)


DATASET_FACTORY = {
    'SimplePolicyPTV3AdaNorm': (SimplePolicyDataset, ptv3_collate_fn),
    'SimplePolicyPTV3CA': (SimplePolicyDataset, ptv3_collate_fn),
    'SimplePolicyPTV3Concat': (SimplePolicyDataset, ptv3_collate_fn),
}

MODEL_FACTORY = {
    'SimplePolicyPTV3AdaNorm': SimplePolicyPTV3AdaNorm,
    'SimplePolicyPTV3CA': SimplePolicyPTV3CA,
    'SimplePolicyPTV3Concat': SimplePolicyPTV3Concat,
}


def main(config):
    config.defrost()
    default_gpu, n_gpu, device = set_cuda(config)

    if default_gpu:
        LOGGER.info(
            'device: {} n_gpu: {}, distributed training: {}'.format(
                device, n_gpu, bool(config.local_rank != -1)
            )
        )

    seed = config.SEED
    if config.local_rank != -1:
        seed += config.rank
    set_random_seed(seed)

    # load data training set
    dataset_class, dataset_collate_fn = DATASET_FACTORY[config.MODEL.model_class]
    trn_dataset = dataset_class(**config.TRAIN_DATASET)
    LOGGER.info(f'#num_train: {len(trn_dataset)}')
    trn_dataloader, pre_epoch = build_dataloader(
        trn_dataset, dataset_collate_fn, True, config
    )

    if config.VAL_DATASET.use_val:
        val_dataset = dataset_class(**config.VAL_DATASET)
        LOGGER.info(f"#num_val: {len(val_dataset)}")
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=config.TRAIN.val_batch_size, shuffle=False,
            num_workers=config.TRAIN.n_workers, pin_memory=True, collate_fn=dataset_collate_fn
        )
    else:
        val_dataloader = None

    LOGGER.info(f'#num_steps_per_epoch: {len(trn_dataloader)}')
    if config.TRAIN.num_train_steps is None:
        config.TRAIN.num_train_steps = len(trn_dataloader) * config.TRAIN.num_epochs
    else:
        # assert config.TRAIN.num_epochs is None, 'cannot set num_train_steps and num_epochs at the same time.'
        config.TRAIN.num_epochs = int(np.ceil(config.TRAIN.num_train_steps / len(trn_dataloader)))
        
    if config.TRAIN.gradient_accumulation_steps > 1:
        config.TRAIN.num_train_steps *= config.TRAIN.gradient_accumulation_steps
        config.TRAIN.num_epochs *= config.TRAIN.gradient_accumulation_steps

    # setup loggers
    if default_gpu:
        save_training_meta(config)
        # TB_LOGGER.create(os.path.join(config.output_dir, 'logs'))
        if config.tfboard_log_dir is None:
            output_dir_tokens = config.output_dir.split('/')
            config.tfboard_log_dir = os.path.join(output_dir_tokens[0], 'TFBoard', *output_dir_tokens[1:])
        TB_LOGGER.create(config.tfboard_log_dir)
        model_saver = ModelSaver(os.path.join(config.output_dir, 'ckpts'))
        add_log_to_file(os.path.join(config.output_dir, 'logs', 'log.txt'))
    else:
        LOGGER.disabled = True
        model_saver = NoOp()

    # Prepare model
    model_class = MODEL_FACTORY[config.MODEL.model_class]
    model = model_class(config.MODEL)
    # DDP: SyncBN
    if config.world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Fix parameters
    if config.TRAIN.freeze_params.encoder:
        for param_name, param in model.named_parameters():
            if param_name.startswith('mae_encoder') and 'decoder_block' not in param_name:
                    param.requires_grad = False

    LOGGER.info("Model: nweights %d nparams %d" % (model.num_parameters))
    LOGGER.info("Model: trainable nweights %d nparams %d" % (model.num_trainable_parameters))
    
    config.freeze()

    # Load from checkpoint
    model_checkpoint_file = config.checkpoint
    optimizer_checkpoint_file = os.path.join(
        config.output_dir, 'ckpts', 'train_state_latest.pt'
    )
    if os.path.exists(optimizer_checkpoint_file) and config.TRAIN.resume_training:
        LOGGER.info('Load the optimizer checkpoint from %s' % optimizer_checkpoint_file)
        optimizer_checkpoint = torch.load(
            optimizer_checkpoint_file, map_location=lambda storage, loc: storage
        )
        lastest_model_checkpoint_file = os.path.join(
            config.output_dir, 'ckpts', 'model_step_%d.pt' % optimizer_checkpoint['step']
        )
        if os.path.exists(lastest_model_checkpoint_file):
            LOGGER.info('Load the model checkpoint from %s' % lastest_model_checkpoint_file)
            model_checkpoint_file = lastest_model_checkpoint_file
        global_step = optimizer_checkpoint['step']
        restart_epoch = global_step // len(trn_dataloader)
    else:
        optimizer_checkpoint = None
        # to compute training statistics
        restart_epoch = 0
        global_step = restart_epoch * len(trn_dataloader) 

    if model_checkpoint_file is not None:
        checkpoint = torch.load(
            model_checkpoint_file, map_location=lambda storage, loc: storage)
        LOGGER.info('Load the model checkpoint (%d params)' % len(checkpoint))
        new_checkpoint = {}
        state_dict = model.state_dict()
        for k, v in checkpoint.items():
            if k in state_dict:
                # TODO: mae_encoder.encoder.first_conv.0.weight
                if k == 'mae_encoder.encoder.first_conv.0.weight':
                    if v.size(1) != state_dict[k].size(1):
                        new_checkpoint[k] = torch.zeros_like(state_dict[k])
                        min_v_size = min(v.size(1), state_dict[k].size(1))
                        new_checkpoint[k][:, :min_v_size] = v[:, :min_v_size]
                if v.size() == state_dict[k].size():
                    if config.TRAIN.resume_encoder_only and (k.startswith('mae_decoder') or 'decoder_block' in k):
                        continue
                    new_checkpoint[k] = v
        LOGGER.info('Resumed the model checkpoint (%d params)' % len(new_checkpoint))
        model.load_state_dict(new_checkpoint, strict=config.checkpoint_strict_load)

    model.train()
    # set_dropout(model, config.TRAIN.dropout)
    model = wrap_model(model, device, config.local_rank, find_unused_parameters=True)

    # Prepare optimizer
    optimizer, init_lrs = build_optimizer(model, config.TRAIN)
    if optimizer_checkpoint is not None:
        optimizer.load_state_dict(optimizer_checkpoint['optimizer'])

    if default_gpu:
        pbar = tqdm(initial=global_step, total=config.TRAIN.num_train_steps)
    else:
        pbar = NoOp()

    LOGGER.info(f"***** Running training with {config.world_size} GPUs *****")
    LOGGER.info("  Batch size = %d", config.TRAIN.train_batch_size if config.local_rank == -1 
                else config.TRAIN.train_batch_size * config.world_size)
    LOGGER.info("  Accumulate steps = %d", config.TRAIN.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", config.TRAIN.num_train_steps)

    optimizer.zero_grad()
    optimizer.step()

    running_metrics = {}

    best_val_step, best_val_metric = None, np.inf
    
    for epoch_id in range(restart_epoch, config.TRAIN.num_epochs):
        if global_step >= config.TRAIN.num_train_steps:
            break

        # In distributed mode, calling the set_epoch() method at the beginning of each epoch
        pre_epoch(epoch_id)
        
        for step, batch in enumerate(trn_dataloader):
            # forward pass
            _, losses = model(batch, compute_loss=True, compute_final_action=False)

            # backward pass
            if config.TRAIN.gradient_accumulation_steps > 1:  # average loss
                losses['total'] = losses['total'] / config.TRAIN.gradient_accumulation_steps
            losses['total'].backward()

            for key, value in losses.items():
                TB_LOGGER.add_scalar(f'step/loss_{key}', value.item(), global_step)
                running_metrics.setdefault(f'loss_{key}', RunningMeter(f'loss_{key}'))
                running_metrics[f'loss_{key}'](value.item())

            # optimizer update and logging
            if (step + 1) % config.TRAIN.gradient_accumulation_steps == 0:
                global_step += 1
                # learning rate scheduling
                lr_decay_rate = get_lr_sched_decay_rate(global_step, config.TRAIN)
                for kp, param_group in enumerate(optimizer.param_groups):
                    param_group['lr'] = lr_this_step = max(init_lrs[kp] * lr_decay_rate, 1e-8)
                TB_LOGGER.add_scalar('lr', lr_this_step, global_step)

                # log loss
                # NOTE: not gathered across GPUs for efficiency
                TB_LOGGER.step()

                # update model params
                if config.TRAIN.grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.TRAIN.grad_norm
                    )
                    TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
                optimizer.step()
                optimizer.zero_grad()
                pbar.update(1)

            if global_step % config.TRAIN.log_steps == 0:
                # monitor training throughput
                LOGGER.info(
                    f'==============Epoch {epoch_id} Step {global_step}===============')
                LOGGER.info(', '.join(['%s:%.4f' % (lk, lv.val) for lk, lv in running_metrics.items()]))
                LOGGER.info('===============================================')                

            if global_step % config.TRAIN.save_steps == 0:
                model_saver.save(model, global_step, optimizer=optimizer, rewrite_optimizer=True)

            if (val_dataloader is not None) and (global_step % config.TRAIN.val_steps == 0):
                val_metrics = validate(model, val_dataloader)
                LOGGER.info(f'=================Validation=================')
                metric_str = ', '.join(['%s: %.4f' % (lk, lv) for lk, lv in val_metrics.items()])
                LOGGER.info(metric_str)
                LOGGER.info('===============================================')
                if val_metrics['pos_loss'] < best_val_metric:
                    best_val_metric = val_metrics['pos_loss']
                    best_val_step = global_step
                model.train()

            if global_step >= config.TRAIN.num_train_steps:
                break

    if global_step % config.TRAIN.save_steps != 0:
        LOGGER.info(
            f'==============Epoch {epoch_id} Step {global_step}===============')
        LOGGER.info(', '.join(['%s:%.4f' % (lk, lv.val) for lk, lv in running_metrics.items()]))
        LOGGER.info('===============================================')
        model_saver.save(model, global_step, optimizer=optimizer, rewrite_optimizer=True)

        val_metrics = validate(model, val_dataloader)
        LOGGER.info(f'=================Validation=================')
        metric_str = ', '.join(['%s: %.4f' % (lk, lv) for lk, lv in val_metrics.items()])
        LOGGER.info(metric_str)
        LOGGER.info('===============================================')
        if val_metrics['pos_loss'] < best_val_metric:
            best_val_metric = val_metrics['pos_loss']
            best_val_step = global_step

    LOGGER.info(
        f'Validation: Best loss: {best_val_metric:.4f} at step {best_val_step}'
    )

@torch.no_grad()
def validate(model, val_dataloader):
    model.eval()
    pos_loss, rot_loss, open_loss, total_loss, num_examples, num_batches = 0, 0, 0, 0, 0, 0
    pos_l1_loss, open_acc = 0, 0
    for batch in val_dataloader:
        pred_action, loss = model(batch, compute_loss=True)
        pred_action = pred_action.cpu()
        pred_open = torch.sigmoid(pred_action[..., -1]) > 0.5
        open_acc += (pred_open == batch['gt_actions'][..., -1].cpu()).float().sum().item()
        pos_l1_loss += F.l1_loss(pred_action[..., :3], batch['gt_actions'][..., :3].cpu())
        if 'layer_11' in loss:
            pos_loss += loss['layer_11_pos'].item()
            total_loss += loss['layer_11'].item() 
        else:
            pos_loss += loss['pos'].item()
            rot_loss += loss['rot'].item()
            open_loss += loss['open'].item()
            total_loss += loss['total'].item() 
        num_examples += pred_action.size(0)
        num_batches += 1
        
    return {
        'total_loss': total_loss / num_batches, 
        'pos_loss': pos_loss / num_batches,
        'pos_l1_loss': pos_l1_loss / num_batches,
        'rot_loss': rot_loss / num_batches,
        'open_loss': open_loss / num_batches,
        'open_acc': open_acc / num_examples, 
    }


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line (use , to separate values in a list)",
    )
    args = parser.parse_args()

    config = get_config(args.exp_config, args.opts)

    if os.path.exists(config.output_dir) and os.listdir(config.output_dir):
        LOGGER.warning(
            "Output directory ({}) already exists and is not empty.".format(
                config.output_dir
            )
        )

    return config


if __name__ == '__main__':
    config = build_args()
    main(config)
