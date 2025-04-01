import os
import sys

import json
import numpy as np
import copy

import open3d as o3d

import lmdb
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

from tqdm import tqdm
import argparse

from genrobo3d.configs.rlbench.constants import get_robot_workspace


def generate_action_trajectories(actions, new_keystep_ids, sep_open_keystep_ids=None):
    num_steps = len(actions)
    sep_ids = set()
    if sep_open_keystep_ids is not None:
        for t in sep_open_keystep_ids:
            if t < 0:
                sep_ids.add(num_steps + t)
            else:
                sep_ids.add(t)

    traj_ids, trajs, end_open_actions = [], [], []
    for step_sidx, step_eidx in zip(new_keystep_ids[:-1], new_keystep_ids[1:]):
        if step_eidx == -1:
            step_eidx = num_steps - 1
        traj_ids.append(np.arange(step_sidx+1, step_eidx+1))
        traj = copy.deepcopy(actions[step_sidx+1: step_eidx+1])

        # separate gripper openness at certain steps
        if step_eidx in sep_ids:
            if traj[-1][-1] != 1:
                print('last action is not open', traj[-1][-1])
            if actions[step_eidx-1][-1] != 0:
                print('previous action is already open', actions[step_eidx-1][-1])
            traj[-1][-1] = actions[step_eidx-1][-1]
            end_open_actions.append(True)
        else:
            end_open_actions.append(False)
        trajs.append(traj)

    return traj_ids, trajs, end_open_actions
    
def expand_action_trajectories(traj_ids, trajs, end_open_actions):
    new_trajs, new_end_open_actions, is_new_keystep = [], [], []
    for traj_id, trajs, end_open_action in zip(traj_ids, trajs, end_open_actions):
        for i in range(len(traj_id)):
            new_trajs.append(trajs[i:])
            new_end_open_actions.append(end_open_action)
            if i == 0:
                is_new_keystep.append(True)
            else:
                is_new_keystep.append(False)
    new_trajs.append([])
    new_end_open_actions.append(False)
    is_new_keystep.append(False)
    return new_trajs, new_end_open_actions, is_new_keystep


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--old_keystep_pcd_dir', required=True)
    parser.add_argument('--new_keystep_pcd_dir', required=True)
    args = parser.parse_args()

    old_keystep_dir = args.old_keystep_pcd_dir
    new_keystep_dir = args.new_keystep_pcd_dir
    os.makedirs(new_keystep_dir, exist_ok=True)

    asset_dir = os.path.join(os.environ.get('HOME'), 'codes/genrobot3d/assets')

    tmp = json.load(open(os.path.join(asset_dir, 'task_new_keystep_ids.json')))
    old_num_keysteps = tmp['old_num_keysteps']
    new_keystep_ids = tmp['new_keystep_ids']
    sep_open_keystep_ids = tmp['separate_gripper_open_at_old_keystep']

    taskvars = json.load(open(os.path.join(asset_dir, 'taskvars_train.json')))
    print('#taskvars', len(taskvars))

    TABLE_HEIGHT = get_robot_workspace()['TABLE_HEIGHT']

    for taskvar in tqdm(taskvars):
        task, variation = taskvar.split('+')

        task_num_keysteps = old_num_keysteps[task]
        task_new_keystep_ids = new_keystep_ids[task]

        out_lmdb_dir = os.path.join(new_keystep_dir, taskvar)
        if os.path.exists(out_lmdb_dir):
            print(taskvar, 'existed!')
            continue

        num_invalid_episodes = 0
        out_lmdb_env = lmdb.open(out_lmdb_dir, map_size=int(1024**4))
        with lmdb.open(os.path.join(old_keystep_dir, taskvar), readonly=True) as lmdb_env:
            with lmdb_env.begin() as txn:
                for episode_key, value in txn.cursor():
                    # episode_key = episode_key.decode('ascii')
                    
                    value = msgpack.unpackb(value)

                    if len(value['key_frameids']) not in task_num_keysteps:
                        num_invalid_episodes += 1
                        continue

                    new_value = {
                        'xyz': [], 'rgb': [], 'sem': [],
                        'ee_pose': value['action'],
                        'bbox_info': value['bbox_info'],
                        'pose_info': value['pose_info'],          
                    }

                    for t in range(len(value['key_frameids'])):
                        xyz = value['xyz'][t]
                        rgb = value['rgb'][t]
                        sem = value['sem'][t]

                        # remove table (already removed background)
                        mask = xyz[:, 2] > TABLE_HEIGHT
                        xyz = xyz[mask]
                        rgb = rgb[mask]
                        sem = sem[mask]

                        new_value['xyz'].append(xyz)
                        new_value['rgb'].append(rgb)
                        new_value['sem'].append(sem)

                    traj_ids, trajs, end_open_actions = generate_action_trajectories(
                        value['action'], task_new_keystep_ids, sep_open_keystep_ids.get(task, None)
                    )
                    new_value['trajs'], new_value['end_open_actions'], new_value['is_new_keystep'] = expand_action_trajectories(
                        traj_ids, trajs, end_open_actions
                    )
                    assert len(new_value['trajs']) == len(value['action'])

                    out_txn = out_lmdb_env.begin(write=True)
                    out_txn.put(episode_key, msgpack.packb(new_value))
                    out_txn.commit()

        out_lmdb_env.close()

        print(taskvar, '#invalid epsiodes', num_invalid_episodes)


if __name__ == '__main__':
    main()
