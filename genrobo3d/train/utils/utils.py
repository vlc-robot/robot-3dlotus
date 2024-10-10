import os
from pathlib import Path

import numpy as np
import random
import torch

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_expr_dirs(output_dir):
    log_dir = os.path.join(output_dir, 'logs')
    ckpt_dir = os.path.join(output_dir, 'ckpts')
    pred_dir = os.path.join(output_dir, 'preds')
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    return log_dir, ckpt_dir, pred_dir

