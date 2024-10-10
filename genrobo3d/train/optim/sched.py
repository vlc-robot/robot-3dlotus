"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

optimizer learning rate scheduling helpers
"""
import math

def noam_schedule(step, warmup_step=4000):
    """ original Transformer schedule"""
    if step <= warmup_step:
        return step / warmup_step
    return (warmup_step ** 0.5) * (step ** -0.5)


def warmup_linear(step, warmup_step, tot_step):
    """ BERT schedule """
    if step < warmup_step:
        return step / warmup_step
    return max(0, (tot_step-step)/(tot_step-warmup_step))

def warmup_inverse_sqrt(step, warmup_step, tot_step):
    """Decay the LR based on the inverse square root of the update number.
    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (``--warmup-init-lr``) until the configured
    learning rate (``--lr``). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.

    During warmup::

      lrs = torch.linspace(cfg.warmup_init_lr, cfg.lr, cfg.warmup_updates)
      lr = lrs[update_num]

    After warmup::

      decay_factor = cfg.lr * sqrt(cfg.warmup_updates)
      lr = decay_factor / sqrt(update_num)
    """
    if step < warmup_step:
        return step / warmup_step
    else:
        return warmup_step**0.5 * step**-0.5

def warmup_cosine(
    step: int, warmup_step: int, tot_step: int, num_cycles: float = 0.5
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    """
    if step < warmup_step:
        return step / warmup_step
    progress = float(step - warmup_step) / float(max(1, tot_step - warmup_step))
    return 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))

def warmup_cosine_hard_restarts(
    step: int, warmup_step: int, tot_step: int, num_cycles: int
):
    if step < warmup_step:
        return step / warmup_step
    progress = float(step - warmup_step) / float(max(1, tot_step - warmup_step))
    if progress >= 1.0:
        return 0.0
    return 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0)))

def warmup_stepwise(
    step: int, warmup_step: int, tot_step: int, step_size: int, step_gamma: float
):
    if step < warmup_step:
        return step / warmup_step
    rate = step_gamma ** (step // step_size)
    return rate
    
def get_lr_sched(global_step, opts):
    fn_args = [global_step, opts.warmup_steps, opts.num_train_steps]
    # learning rate scheduling
    if opts.lr_sched == 'linear':
        func = warmup_linear
    elif opts.lr_sched == 'inverse_sqrt':
        func = warmup_inverse_sqrt
    elif opts.lr_sched == 'cosine':
        func = warmup_cosine
    elif opts.lr_sched == 'cosine_cycle':
        func = warmup_cosine_hard_restarts
        fn_args.append(opts.num_cosine_cycles)
    else:
        raise NotImplementedError(f'invalid lr scheduler {opts.lr_sched}')

    lr_this_step = opts.learning_rate * func(*fn_args)
    if lr_this_step <= 0:
        lr_this_step = 1e-8
    return lr_this_step

def get_lr_sched_decay_rate(global_step, opts):
    fn_args = [global_step, opts.warmup_steps, opts.num_train_steps]
    # learning rate scheduling
    if opts.lr_sched == 'linear':
        lr_decay_fn = warmup_linear
    elif opts.lr_sched == 'inverse_sqrt':
        lr_decay_fn = warmup_inverse_sqrt
    elif opts.lr_sched == 'cosine':
        lr_decay_fn = warmup_cosine
    elif opts.lr_sched == 'cosine_cycle':
        lr_decay_fn = warmup_cosine_hard_restarts
        fn_args.append(opts.num_cosine_cycles)
    elif opts.lr_sched == 'stepwise':
        lr_decay_fn = warmup_stepwise
        fn_args.extend([opts.lr_decay_step_size, opts.lr_decay_gamma])

    lr_decay_rate = lr_decay_fn(*fn_args)
    lr_decay_rate = max(lr_decay_rate, 1e-5)
    return lr_decay_rate
