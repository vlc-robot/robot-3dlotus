"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Misc lr helper
"""
from torch.optim import Adam, Adamax

from .adamw import AdamW
from .rangerlars import RangerLars


def build_optimizer(model, opts):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    rgb_enc_params, other_params = {}, {}
    for n, p in param_optimizer:
        if not p.requires_grad: continue
        if 'rgb_encoder' in n:
            rgb_enc_params[n] = p
        else:
            other_params[n] = p
    
    optimizer_grouped_parameters = []
    init_lrs = []
    for ptype, pdict in [('rgb', rgb_enc_params), ('others', other_params)]:
        if len(pdict) == 0:
            continue
        init_lr = opts.learning_rate
        if ptype == 'rgb':
            init_lr = init_lr * getattr(opts, 'rgb_encoder_lr_multi', 1)
        optimizer_grouped_parameters.extend([
            {'params': [p for n, p in pdict.items()
                        if not any(nd in n for nd in no_decay)],
            'weight_decay': opts.weight_decay, 'lr': init_lr},
            {'params': [p for n, p in pdict.items()
                        if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0, 'lr': init_lr}
        ])
        init_lrs.extend([init_lr] * 2)
        
    # currently Adam only
    if opts.optim == 'adam':
        OptimCls = Adam
    elif opts.optim == 'adamax':
        OptimCls = Adamax
    elif opts.optim == 'adamw':
        OptimCls = AdamW
    elif opts.optim == 'rangerlars':
        OptimCls = RangerLars
    else:
        raise ValueError('invalid optimizer')
    optimizer = OptimCls(optimizer_grouped_parameters,
                         lr=opts.learning_rate, betas=opts.betas)
    return optimizer, init_lrs
