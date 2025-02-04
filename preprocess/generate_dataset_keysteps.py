from typing import Tuple, Dict, List

import os
import argparse
import numpy as np
from tqdm import tqdm
import collections
from PIL import Image
import glob

import lmdb
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

from genrobo3d.utils.rlbench_keystep_detection import keypoint_discovery
from genrobo3d.rlbench.coord_transforms import convert_gripper_pose_world_to_image

from genrobo3d.rlbench.environments import RLBenchEnv


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--microstep_data_dir', required=True, help="data/train_dataset/microsteps/seed0")
    parser.add_argument('--keystep_data_dir', required=True, help="data/train_dataset/keysteps/seed0")
    parser.add_argument('--task', required=True, type=str)
    parser.add_argument('--variation_id', required=True, type=int)
    parser.add_argument(
        '--cameras', nargs='+', choices=["left_shoulder", "right_shoulder", "wrist", "front"],
        default=["left_shoulder", "right_shoulder", "wrist", "front"]
    )
    parser.add_argument('--save_masks', action='store_true', default=False)
    parser.add_argument('--image_size', type=int, default=128)
    args = parser.parse_args()
    return args


def get_observation(task_str: str, variation: int, episode: int, env: RLBenchEnv):
    demo = env.get_demo(task_str, variation, episode)

    key_frames = keypoint_discovery(demo)

    # HACK for tower3
    if task_str == "tower3":
        key_frames = [k for i, k in enumerate(key_frames) if i % 6 in set([1, 4])]
    # HACK tower4
    # elif task_str == "tower4":
    #     key_frames = key_frames[6:]

    key_frames.insert(0, 0)

    state_dict_ls = collections.defaultdict(list)
    for f in key_frames:
        state_dict = env.get_observation(demo._observations[f])
        for k, v in state_dict.items():
            if len(v) > 0:
                if k == 'arm_links_info':
                    for arm_link_key, arm_link_value in state_dict[k][0].items():
                        state_dict_ls[arm_link_key].append(arm_link_value)
                    for arm_link_key, arm_link_value in state_dict[k][1].items():
                        state_dict_ls[arm_link_key].append(arm_link_value)
                else:
                    # rgb: (num_of_cameras, H, W, C); gripper: (7+1, )
                    state_dict_ls[k].append(v)

    for k, v in state_dict_ls.items():
        state_dict_ls[k] = np.stack(v, 0)  # (T, N, H, W, C)

    action_ls = state_dict_ls["gripper"]  # (T, 7+1)
    del state_dict_ls["gripper"]

    return demo, key_frames, state_dict_ls, action_ls


def generate_keystep_dataset(args):
    # load RLBench environment
    rlbench_env = RLBenchEnv(
        data_path=args.microstep_data_dir,
        apply_rgb=True,
        apply_depth=True,
        apply_pc=True,
        apply_mask=args.save_masks,
        apply_cameras=args.cameras,
        image_size=[args.image_size, args.image_size],
    )

    task_str = args.task
    variation_id = args.variation_id

    episodes_dir = os.path.join(
        args.microstep_data_dir, task_str, f"variation{variation_id}", "episodes"
    )

    if not os.path.exists(episodes_dir):
        print(f"{episodes_dir} does not exist, skip")
        return

    output_dir = os.path.join(args.keystep_data_dir, f"{task_str}+{variation_id}")
    os.makedirs(output_dir, exist_ok=True)

    lmdb_env = lmdb.open(output_dir, map_size=int(1024**4))

    for ep in tqdm(glob.glob(os.path.join(episodes_dir, "episode*"))):
        if not os.path.exists(os.path.join(ep, "low_dim_obs.pkl")):
            continue
        episode = int(os.path.basename(ep)[7:])
        try:
            demo, key_frameids, state_dict_ls, action_ls = get_observation(
                task_str, variation_id, episode, rlbench_env
            )
        except (FileNotFoundError, RuntimeError, IndexError) as e:
            print(e)
            continue

        gripper_pose = []
        for key_frameid in key_frameids:
            gripper_pose.append(
                {
                    cam: convert_gripper_pose_world_to_image(demo[key_frameid], cam)
                    for cam in args.cameras
                }
            )
        bbox_info = {}
        pose_info = {}
        for state_k in state_dict_ls:
            if "bbox" in state_k:
                bbox_info[state_k] = state_dict_ls[state_k]
            if "pose" in state_k:
                pose_info[state_k] = state_dict_ls[state_k]
        
        outs = {
            "key_frameids": key_frameids,
            "rgb": state_dict_ls["rgb"],  # (T, N, H, W, 3)
            "pc": state_dict_ls["pc"],  # (T, N, H, W, 3)
            "depth": state_dict_ls["depth"],  # (T, N, H, W)
            "action": action_ls,  # (T, A)
            "gripper_pose": gripper_pose,  # [T of dict]
            "bbox_info": bbox_info,
            "pose_info": pose_info,
        }

        if args.save_masks:
            outs["mask"] = state_dict_ls["gt_mask"]  # (T, N, H, W)

        txn = lmdb_env.begin(write=True)
        txn.put(f"episode{episode}".encode("ascii"), msgpack.packb(outs))
        txn.commit()

    lmdb_env.close()


if __name__ == "__main__":
    args = build_parser()
    generate_keystep_dataset(args)
