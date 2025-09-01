"""
module load singularity

conda activate gondola

sif_image=$SINGULARITY_ALLOWED_DIR/nvcuda_v2.sif
export python_bin=$HOME/miniconda3/envs/gondola/bin/python
export SINGULARITYENV_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${COPPELIASIM_ROOT}

singularity exec --bind $HOME:$HOME,$WORK:$WORK,$SCRATCH:$SCRATCH --nv ${sif_image} \
    xvfb-run -a ${python_bin} preprocess/evaluate_rlbench_microsteps.py \
    --microstep_data_dir data/gembench/train_dataset/microsteps/seed0
"""

from typing import Tuple, Dict, List

import os
import numpy as np
import random

from pathlib import Path
from tqdm import tqdm
import collections
import tap
import jsonlines
import glob

import pickle as pkl

from genrobo3d.rlbench.environments import RLBenchEnv


class Arguments(tap.Tap):
    microstep_data_dir: Path = "data/gembench/train_dataset/microsteps/seed0"

    seed: int = 0

    cameras: Tuple[str, ...] = ("left_shoulder", "right_shoulder", "wrist")
    max_tries: int = 10
    max_steps: int = 1000


class MicrostepActioner(object):
    def __init__(self, microstep_data_dir) -> None:
        self.microstep_data_dir = microstep_data_dir
        self.actions = None

    def predict(self, task_str, variation, step_id, obs_state_dict, episode_id, instructions):
        out = {}
        if step_id == 0:
            low_dim_obs = pkl.load(
                open(os.path.join(self.microstep_data_dir, task_str, f'variation{variation}', 'episodes', episode_id, 'low_dim_obs.pkl'), 'rb')
            )
            self.actions = [
                np.hstack([x.gripper_pose, x.gripper_open]) for x in low_dim_obs[1:]
            ]
            
        if step_id < len(self.actions):
            out['action'] = self.actions[step_id]
        else:
            # Sometimes, RLBench motion planner does not bring the gripper to the target pose, need to run it multiple times
            out['action'] = np.zeros((8, ), dtype=np.float32)
            # out['action'] = self.actions[-1]
        return out


def evaluate_keysteps(args):
    np.random.seed(args.seed)
    random.seed(args.seed)

    env = RLBenchEnv(
        data_path=args.microstep_data_dir,
        apply_rgb=True,
        apply_pc=True,
        apply_cameras=args.cameras,
        headless=True,
    )

    actioner = MicrostepActioner(args.microstep_data_dir)

    taskvar_dirs = glob.glob(f'{args.microstep_data_dir}/*/*')
    result_file = os.path.join(args.microstep_data_dir, 'taskvar_srs.jsonl')

    for taskvar_dir in taskvar_dirs:
        task_str, variation = taskvar_dir.split('/')[-2:]
        variation = int(variation[len('variation'):])
        taskvar = f'{task_str}+{variation}'

        episode_dir = os.path.join(taskvar_dir, "episodes")
        episode_ids = os.listdir(episode_dir)
        episode_ids.sort(key=lambda ep: int(ep[7:]))

        demo_keys, demos = [], []
        for idx, ep in enumerate(episode_ids):
            # episode_id = int(ep[7:])
            try:
                demo = env.get_demo(task_str, variation, idx, load_images=False)
                demo_keys.append(f'episode{idx}')
                demos.append(demo)
            except Exception as e:
                print('\tProblem to load demo_id:', idx, ep)
                print(e)
 
        success_rate = env.evaluate(
            task_str, variation,
            max_episodes=args.max_steps,
            num_demos=len(demos),
            log_dir=None,
            actioner=actioner,
            demos=demos,
            demo_keys=demo_keys,
            max_tries=args.max_tries,
            save_image=False,
            record_video=False,
            include_robot_cameras=False,
            video_rotate_cam=False,
            video_resolution=False,
        )

        print(taskvar, success_rate * 100)
        with jsonlines.open(result_file, 'a') as outf:
            outf.write({'taskvar': taskvar, 'sr': success_rate})     


if __name__ == '__main__':
    args = Arguments().parse_args()
    evaluate_keysteps(args)