"""
Modified from RLBench/tools/dataset_generator.py
"""

import os
import argparse
import pickle
import json
from PIL import Image
import numpy as np
import random

from tqdm import tqdm
from pyrep.const import RenderMode
from pathlib import Path

from rlbench import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment
import rlbench.backend.task as task
from rlbench.backend import utils
from rlbench.backend.const import *


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--microstep_data_dir', required=True, help='Where to save the demos.')
    parser.add_argument('--task', required=True, help='The task to collect.')
    parser.add_argument('--variation_id', required=True, default=0, type=int, help='Variation_id to collect for the task.')
    parser.add_argument('--seed', type=int, default=0, help="Seed of randomness")

    parser.add_argument('--image_size', type=int, default=128, help="The size of the images tp save.")
    parser.add_argument(
        '--renderer', default='opengl3', choices=['opengl', 'opengl3'],
        help="The renderer to use. opengl does not include shadows, but is faster",
    )
    parser.add_argument('--episodes_per_task', type=int, default=10, help="The number of episodes to collect per task.")
    parser.add_argument('--max_len', default=500, type=int, help="Crop episodes longer than max_len.")

    parser.add_argument('--live_demos', action='store_true', default=False, help="Live demo or loading microsteps")
    parser.add_argument('--prev_state_dir', default='', help='Recreate datasets given the states of demos')
    args = parser.parse_args()
    return args


def save_demo(demo, example_path):
    # Save image data first, and then None the image data, and pickle
    left_shoulder_rgb_path = os.path.join(example_path, LEFT_SHOULDER_RGB_FOLDER)
    left_shoulder_depth_path = os.path.join(example_path, LEFT_SHOULDER_DEPTH_FOLDER)
    left_shoulder_mask_path = os.path.join(example_path, LEFT_SHOULDER_MASK_FOLDER)
    right_shoulder_rgb_path = os.path.join(example_path, RIGHT_SHOULDER_RGB_FOLDER)
    right_shoulder_depth_path = os.path.join(example_path, RIGHT_SHOULDER_DEPTH_FOLDER)
    right_shoulder_mask_path = os.path.join(example_path, RIGHT_SHOULDER_MASK_FOLDER)
    overhead_rgb_path = os.path.join(example_path, OVERHEAD_RGB_FOLDER)
    overhead_depth_path = os.path.join(example_path, OVERHEAD_DEPTH_FOLDER)
    overhead_mask_path = os.path.join(example_path, OVERHEAD_MASK_FOLDER)
    wrist_rgb_path = os.path.join(example_path, WRIST_RGB_FOLDER)
    wrist_depth_path = os.path.join(example_path, WRIST_DEPTH_FOLDER)
    wrist_mask_path = os.path.join(example_path, WRIST_MASK_FOLDER)
    front_rgb_path = os.path.join(example_path, FRONT_RGB_FOLDER)
    front_depth_path = os.path.join(example_path, FRONT_DEPTH_FOLDER)
    front_mask_path = os.path.join(example_path, FRONT_MASK_FOLDER)

    os.makedirs(left_shoulder_rgb_path, exist_ok=True)
    os.makedirs(left_shoulder_depth_path, exist_ok=True)
    os.makedirs(left_shoulder_mask_path, exist_ok=True)
    os.makedirs(right_shoulder_rgb_path, exist_ok=True)
    os.makedirs(right_shoulder_depth_path, exist_ok=True)
    os.makedirs(right_shoulder_mask_path, exist_ok=True)
    os.makedirs(overhead_rgb_path, exist_ok=True)
    os.makedirs(overhead_depth_path, exist_ok=True)
    os.makedirs(overhead_mask_path, exist_ok=True)
    os.makedirs(wrist_rgb_path, exist_ok=True)
    os.makedirs(wrist_depth_path, exist_ok=True)
    os.makedirs(wrist_mask_path, exist_ok=True)
    os.makedirs(front_rgb_path, exist_ok=True)
    os.makedirs(front_depth_path, exist_ok=True)
    os.makedirs(front_mask_path, exist_ok=True)

    for i, obs in enumerate(demo):
        left_shoulder_rgb = Image.fromarray(obs.left_shoulder_rgb)
        left_shoulder_depth = utils.float_array_to_rgb_image(
            obs.left_shoulder_depth, scale_factor=DEPTH_SCALE
        )
        left_shoulder_mask = Image.fromarray(
            (obs.left_shoulder_mask * 255).astype(np.uint8)
        )
        right_shoulder_rgb = Image.fromarray(obs.right_shoulder_rgb)
        right_shoulder_depth = utils.float_array_to_rgb_image(
            obs.right_shoulder_depth, scale_factor=DEPTH_SCALE
        )
        right_shoulder_mask = Image.fromarray(
            (obs.right_shoulder_mask * 255).astype(np.uint8)
        )
        overhead_rgb = Image.fromarray(obs.overhead_rgb)
        overhead_depth = utils.float_array_to_rgb_image(
            obs.overhead_depth, scale_factor=DEPTH_SCALE
        )
        overhead_mask = Image.fromarray((obs.overhead_mask * 255).astype(np.uint8))
        wrist_rgb = Image.fromarray(obs.wrist_rgb)
        wrist_depth = utils.float_array_to_rgb_image(
            obs.wrist_depth, scale_factor=DEPTH_SCALE
        )
        wrist_mask = Image.fromarray((obs.wrist_mask * 255).astype(np.uint8))
        front_rgb = Image.fromarray(obs.front_rgb)
        front_depth = utils.float_array_to_rgb_image(
            obs.front_depth, scale_factor=DEPTH_SCALE
        )
        front_mask = Image.fromarray((obs.front_mask * 255).astype(np.uint8))

        left_shoulder_rgb.save(os.path.join(left_shoulder_rgb_path, IMAGE_FORMAT % i))
        left_shoulder_depth.save(
            os.path.join(left_shoulder_depth_path, IMAGE_FORMAT % i)
        )
        left_shoulder_mask.save(os.path.join(left_shoulder_mask_path, IMAGE_FORMAT % i))
        right_shoulder_rgb.save(os.path.join(right_shoulder_rgb_path, IMAGE_FORMAT % i))
        right_shoulder_depth.save(
            os.path.join(right_shoulder_depth_path, IMAGE_FORMAT % i)
        )
        right_shoulder_mask.save(
            os.path.join(right_shoulder_mask_path, IMAGE_FORMAT % i)
        )
        overhead_rgb.save(os.path.join(overhead_rgb_path, IMAGE_FORMAT % i))
        overhead_depth.save(os.path.join(overhead_depth_path, IMAGE_FORMAT % i))
        overhead_mask.save(os.path.join(overhead_mask_path, IMAGE_FORMAT % i))
        wrist_rgb.save(os.path.join(wrist_rgb_path, IMAGE_FORMAT % i))
        wrist_depth.save(os.path.join(wrist_depth_path, IMAGE_FORMAT % i))
        wrist_mask.save(os.path.join(wrist_mask_path, IMAGE_FORMAT % i))
        front_rgb.save(os.path.join(front_rgb_path, IMAGE_FORMAT % i))
        front_depth.save(os.path.join(front_depth_path, IMAGE_FORMAT % i))
        front_mask.save(os.path.join(front_mask_path, IMAGE_FORMAT % i))

        # We save the images separately, so set these to None for pickling.
        obs.left_shoulder_rgb = None
        obs.left_shoulder_depth = None
        obs.left_shoulder_point_cloud = None
        obs.left_shoulder_mask = None
        obs.right_shoulder_rgb = None
        obs.right_shoulder_depth = None
        obs.right_shoulder_point_cloud = None
        obs.right_shoulder_mask = None
        obs.overhead_rgb = None
        obs.overhead_depth = None
        obs.overhead_point_cloud = None
        obs.overhead_mask = None
        obs.wrist_rgb = None
        obs.wrist_depth = None
        obs.wrist_point_cloud = None
        obs.wrist_mask = None
        obs.front_rgb = None
        obs.front_depth = None
        obs.front_point_cloud = None
        obs.front_mask = None

    # Save the low-dimension data
    with open(os.path.join(example_path, LOW_DIM_PICKLE), "wb") as f:
        pickle.dump(demo, f)


def run(task_class, args):

    # Initialise each thread with random seed
    # np.random.seed(None)
    np.random.seed(args.seed)
    random.seed(args.seed)

    img_size = [args.image_size, args.image_size]

    obs_config = ObservationConfig()
    obs_config.set_all(True)
    obs_config.right_shoulder_camera.image_size = img_size
    obs_config.left_shoulder_camera.image_size = img_size
    obs_config.overhead_camera.image_size = img_size
    obs_config.wrist_camera.image_size = img_size
    obs_config.front_camera.image_size = img_size

    # Store depth as 0 - 1
    obs_config.right_shoulder_camera.depth_in_meters = False
    obs_config.left_shoulder_camera.depth_in_meters = False
    obs_config.overhead_camera.depth_in_meters = False
    obs_config.wrist_camera.depth_in_meters = False
    obs_config.front_camera.depth_in_meters = False

    # We want to save the masks as rgb encodings.
    obs_config.left_shoulder_camera.masks_as_one_channel = False
    obs_config.right_shoulder_camera.masks_as_one_channel = False
    obs_config.overhead_camera.masks_as_one_channel = False
    obs_config.wrist_camera.masks_as_one_channel = False
    obs_config.front_camera.masks_as_one_channel = False

    if args.renderer == "opengl":
        obs_config.right_shoulder_camera.render_mode = RenderMode.OPENGL
        obs_config.left_shoulder_camera.render_mode = RenderMode.OPENGL
        obs_config.overhead_camera.render_mode = RenderMode.OPENGL
        obs_config.wrist_camera.render_mode = RenderMode.OPENGL
        obs_config.front_camera.render_mode = RenderMode.OPENGL

    rlbench_env = Environment(
        action_mode=MoveArmThenGripper(JointVelocity(), Discrete()),
        obs_config=obs_config,
        headless=True,
        dataset_root=args.prev_state_dir if not args.live_demos else '',
    )
    rlbench_env.launch()
        
    task_env = rlbench_env.get_task(task_class)
    num_vars = task_env.variation_count()
    print(f'{args.task} total variation counts {num_vars}')
    if args.variation_id >= num_vars:
        print(f'{args.variation_id} is larger than the available variations {num_vars}')
        print(f'Reset to {num_vars-1}')
    variation_id = np.minimum(args.variation_id, num_vars-1)

    task_env.set_variation(variation_id)
    descriptions, obs = task_env.reset()

    variation_path = os.path.join(
        args.microstep_data_dir, task_env.get_name(), VARIATIONS_FOLDER % variation_id
    )
    print('Save to:', variation_path)

    os.makedirs(variation_path, exist_ok=True)

    with open(os.path.join(variation_path, VARIATION_DESCRIPTIONS), "wb") as f:
        pickle.dump(descriptions, f)

    episodes_path = os.path.join(variation_path, EPISODES_FOLDER)
    os.makedirs(episodes_path, exist_ok=True)

    abort_variation = False
    if args.live_demos:
        episodes_per_task = list(range(args.episodes_per_task))
    else:
        episodes_dir = (
            Path(args.prev_state_dir) / task_env.get_name() / f"variation{variation_id}" / "episodes"
        )
        if not os.path.exists(str(episodes_dir)):
            print("taskvar", task_env.get_name(), "not exists")
            print("taskvar", episodes_dir, "not exists")
            return
        episodes_per_task = []
        for ep in tqdm(episodes_dir.glob("episode*")):
            if not (ep/"low_dim_obs.pkl").exists():
                continue
            episode_id = int(ep.stem[7:])
            episodes_per_task.append(episode_id)

    for ex_idx in episodes_per_task:
        print("Task:", task_env.get_name(), "// Variation:", variation_id, "// Demo:", ex_idx)

        attempts = 50
        while attempts > 0:
            episode_path = os.path.join(episodes_path, EPISODE_FOLDER % ex_idx)
            if os.path.exists(episode_path):
                break
            try:
                # TODO: for now we do the explicit looping.
                if args.live_demos:
                    (demo,) = task_env.get_demos(amount=1, live_demos=True)
                else:
                    (demo,) = task_env.get_demos(
                        amount=1,
                        live_demos=False,
                        random_selection=False,
                        from_episode_number=ex_idx,
                        run_loaded_demo=True,
                        load_images=False,
                    )
            except Exception as e:
                attempts -= 1
                if attempts > 0:
                    continue
                problem = (
                    "Failed collecting task %s (variation: %d, "
                    "example: %d). Skipping this task/variation.\n%s\n"
                    % (task_env.get_name(), variation_id, ex_idx, str(e))
                )
                print(problem)
                abort_variation = True
                break
            
            save_demo(demo, episode_path)

            with open(os.path.join(episode_path, VARIATION_DESCRIPTIONS), 'wb') as f:
                pickle.dump(descriptions, f)

            break

        if abort_variation:
            # break
            continue

    rlbench_env.shutdown()


def main(args):

    try:
        task_class = task_file_to_task_class(args.task)
    except:
        print(f'{args.task} does not exist in RLBench')

    os.makedirs(args.microstep_data_dir, exist_ok=True)

    run(task_class, args)    


if __name__ == "__main__":
    args = build_parser()
    main(args)
