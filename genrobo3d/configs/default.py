from typing import List, Optional, Union

import yacs.config

# Default config node
class Config(yacs.config.CfgNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, new_allowed=True)

CN = Config


CONFIG_FILE_SEPARATOR = ';'

# -----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# -----------------------------------------------------------------------------
_C = CN()
_C.SEED = 2023
_C.output_dir = None
_C.checkpoint = None
_C.checkpoint_strict_load = True # true, false
_C.CMD_TRAILING_OPTS = []  # store command line options as list of strings

# -----------------------------------------------------------------------------
# TRAIN
# -----------------------------------------------------------------------------
_C.TRAIN = CN()

_C.TRAIN.resume_training = True # true, false
_C.TRAIN.train_batch_size = 8
_C.TRAIN.gradient_accumulation_steps = 1
_C.TRAIN.num_epochs = None
_C.TRAIN.num_train_steps =  100000
_C.TRAIN.warmup_steps = 5000
_C.TRAIN.log_steps = 1000
_C.TRAIN.save_steps = 5000

_C.TRAIN.optim = 'adamw'
_C.TRAIN.learning_rate = 1e-4
_C.TRAIN.lr_sched = 'cosine' # inverse_sqrt, linear, cosine, cosine_cycle
_C.TRAIN.num_cosine_cycles = None # float
_C.TRAIN.betas = [0.9, 0.98]
_C.TRAIN.weight_decay = 0.05
_C.TRAIN.grad_norm = 5
_C.TRAIN.n_workers = 0
_C.TRAIN.pin_mem = True

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()

# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------
_C.DATASET = CN()


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    :ref:`config_paths` and overwritten by options from :ref:`opts`.

    Args:
        config_paths: List of config paths or string that contains comma
        separated list of config paths.
        opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example, ``opts = ['FOO.BAR',
        0.5]``. Argument can be used for parameter sweeping or quick tests.
    """
    config = _C.clone()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)
            
    if opts:
        config.CMD_TRAILING_OPTS = config.CMD_TRAILING_OPTS + opts
        config.merge_from_list(config.CMD_TRAILING_OPTS)

    config.CMD_TRAILING_OPTS = []

    config.freeze()
    return config
