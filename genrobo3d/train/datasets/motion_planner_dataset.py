import os
import numpy as np
import json
import copy
import random

import lmdb
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

import torch

from scipy.spatial.transform import Rotation as R

from genrobo3d.train.datasets.common import (
    pad_tensors, gen_seq_masks, random_rotate_z
)
from genrobo3d.configs.rlbench.constants import (
    get_rlbench_labels, get_robot_workspace
)
from genrobo3d.utils.rotation_transform import (
    RotationMatrixTransform, quaternion_to_discrete_euler
)
from genrobo3d.utils.robot_box import RobotBox
from genrobo3d.utils.action_position_utils import get_disc_gt_pos_prob
from genrobo3d.train.datasets.simple_policy_dataset import SimplePolicyDataset


class MotionPlannerDataset(SimplePolicyDataset):
    def __init__(
            self, data_dir, action_embed_file, gt_act_obj_label_file,
            taskvar_file=None, num_points=10000, 
            xyz_shift='center', xyz_norm=False, use_height=False,
            max_traj_len=5, pc_label_type='coarse',
            pc_label_augment=False, pc_midstep_augment=False,
            rot_type='quat', instr_embed_type='last', all_step_in_batch=True,
            rm_table=True, rm_robot='none', include_last_step=False, augment_pc=False,
            same_npoints_per_example=False,
            rm_pc_outliers=False, rm_pc_outliers_neighbors=25, euler_resolution=5,
            pos_type='cont', pos_bins=50, pos_bin_size=0.01, 
            pos_heatmap_type='plain', pos_heatmap_no_robot=False, 
            aug_max_rot=45, use_color=False, instr_include_objects=False, 
            real_robot=False, **kwargs
        ):

        assert instr_embed_type in ['last', 'all']
        assert xyz_shift in ['none', 'center', 'gripper']
        assert pos_type in ['cont', 'disc']
        assert rot_type in ['quat', 'rot6d', 'euler', 'euler_disc']
        assert rm_robot in ['none', 'gt', 'box', 'box_keep_gripper']
        assert pc_label_type in ['coarse', 'fine', 'mix']

        self.action_embeds = np.load(action_embed_file, allow_pickle=True).item()
        if instr_embed_type == 'last':
            self.action_embeds = {instr: embeds[-1:] for instr, embeds in self.action_embeds.items()}
        if gt_act_obj_label_file is not None:
            self.gt_act_obj_labels = json.load(open(gt_act_obj_label_file))
        
        if taskvar_file is not None:
            self.taskvars = json.load(open(taskvar_file))
        else:
            self.taskvars = os.listdir(data_dir)

        self.lmdb_envs, self.lmdb_txns = {}, {}
        self.data_ids = []
        for taskvar in self.taskvars:
            if not os.path.exists(os.path.join(data_dir, taskvar)):
                continue
            self.lmdb_envs[taskvar] = lmdb.open(os.path.join(data_dir, taskvar), readonly=True)
            self.lmdb_txns[taskvar] = self.lmdb_envs[taskvar].begin()
            if all_step_in_batch:
                self.data_ids.extend(
                    [(taskvar, key) for key in self.lmdb_txns[taskvar].cursor().iternext(values=False)]
                )
            else:
                for key, value in self.lmdb_txns[taskvar].cursor():
                    value = msgpack.unpackb(value)
                    if pc_midstep_augment:
                        self.data_ids.extend([(taskvar, key, t) for t in range(len(value['xyz']))])
                    else:
                        self.data_ids.extend([(taskvar, key, t) for t in range(len(value['xyz'])) if (value['is_new_keystep'][t]) or (t == len(value['xyz'])-1)])

        self.num_points = num_points
        self.max_traj_len = max_traj_len
        self.pc_label_type = pc_label_type
        self.pc_label_augment = pc_label_augment
        self.pc_midstep_augment = pc_midstep_augment
        self.xyz_shift = xyz_shift
        self.xyz_norm = xyz_norm
        self.use_height = use_height
        self.pos_type = pos_type
        self.rot_type = rot_type
        self.rm_table = rm_table
        self.rm_robot = rm_robot
        self.all_step_in_batch = all_step_in_batch
        self.include_last_step = include_last_step
        self.instr_include_objects = instr_include_objects
        self.use_color = use_color
        self.augment_pc = augment_pc
        self.aug_max_rot = np.deg2rad(aug_max_rot)
        self.rm_pc_outliers = rm_pc_outliers
        self.rm_pc_outliers_neighbors = rm_pc_outliers_neighbors
        self.same_npoints_per_example = same_npoints_per_example
        self.euler_resolution = euler_resolution
        self.pos_bins = pos_bins
        self.pos_bin_size = pos_bin_size
        self.pos_heatmap_type = pos_heatmap_type
        self.pos_heatmap_no_robot = pos_heatmap_no_robot
        self.real_robot = real_robot

        self.TABLE_HEIGHT = get_robot_workspace(real_robot=self.real_robot)['TABLE_HEIGHT']
        self.rotation_transform = RotationMatrixTransform()

    def _get_rotation_from_quat(self, gt_rot):
        if self.rot_type == 'euler':
            gt_rot = self.rotation_transform.quaternion_to_euler(
                torch.from_numpy(gt_rot[None, :]))[0].numpy() / 180.
        elif self.rot_type == 'euler_disc':
            gt_rot = quaternion_to_discrete_euler(gt_rot, self.euler_resolution)
        elif self.rot_type == 'rot6d':
            gt_rot = self.rotation_transform.quaternion_to_ortho6d(
                torch.from_numpy(gt_rot[None, :]))[0].numpy()
        return gt_rot
                
    def _augment_pc(self, xyz, ee_pose, gt_trajs, aug_max_rot):
        # rotate around z-axis
        angle = np.random.uniform(-1, 1) * aug_max_rot
        xyz = random_rotate_z(xyz, angle=angle)
        ee_pose[:3] = random_rotate_z(ee_pose[:3], angle=angle)
        ee_pose[3:-1] = self._rotate_gripper(ee_pose[3:-1], angle)
        for gt_action in gt_trajs:
            gt_action[:3] = random_rotate_z(gt_action[:3], angle=angle)
            gt_action[3:-1] = self._rotate_gripper(gt_action[3:-1], angle)

        # add small noises (+-2mm)
        pc_noises = np.random.uniform(0, 0.002, size=xyz.shape)
        xyz = pc_noises + xyz

        return xyz, ee_pose, gt_trajs
    
    def __getitem__(self, idx):
        if self.all_step_in_batch:
            taskvar, data_id = self.data_ids[idx]
        else:
            taskvar, data_id, data_step = self.data_ids[idx]

        task, variation = taskvar.split('+')

        gt_act_obj_labels = self.gt_act_obj_labels[taskvar]

        data = msgpack.unpackb(self.lmdb_txns[taskvar].get(data_id))

        outs = {
            'data_ids': [], 'pc_fts': [], 'pc_labels': [], 
            'pc_centroids': [], 'pc_radius': [], 'ee_poses': [], 
            'txt_embeds': [], 'gt_trajs': [], 'gt_trajs_stop': [],
        }
        if self.pos_type == 'disc':
            outs['gt_trajs_disc_pos_probs'] = []

        keystep = -1
        num_steps = len(data['xyz'])
        for t in range(num_steps):
            if data['is_new_keystep'][t]:
                keystep += 1

            if (not self.all_step_in_batch) and t != data_step:
                continue
            if (not self.pc_midstep_augment) and (not data['is_new_keystep'][t]) and (t != num_steps - 1):
                continue
            if (not self.include_last_step) and (t == (num_steps - 1)):
                continue

            xyz, rgb, gt_sem = data['xyz'][t], data['rgb'][t], data['sem'][t]
            arm_links_info = (
                {k: v[t] for k, v in data['bbox_info'].items()}, 
                {k: v[t] for k, v in data['pose_info'].items()}
            )

            if t < num_steps - 1:
                gt_traj_len = len(data['trajs'][t])
                gt_trajs = copy.deepcopy(data['trajs'][t])[:self.max_traj_len]
            else:
                gt_traj_len = 1
                gt_trajs = copy.deepcopy(data['trajs'][-2][-1:])
            ee_pose = copy.deepcopy(data['ee_pose'][t])

            action_name = gt_act_obj_labels[keystep]['action']
            if self.instr_include_objects:
                if 'object' in gt_act_obj_labels[keystep]:
                    action_name = f"{action_name} {gt_act_obj_labels[keystep]['object']['name']}"
                if 'target' in gt_act_obj_labels[keystep]:
                    action_name = f"{action_name} to {gt_act_obj_labels[keystep]['target']['name']}"
            action_embed = self.action_embeds[action_name]

            # remove background points
            if self.rm_table:
                mask = xyz[..., 2] > self.TABLE_HEIGHT
                xyz = xyz[mask]
                rgb = rgb[mask]
                gt_sem = gt_sem[mask]
                
            if self.rm_robot.startswith('box'):
                mask = self._get_mask_with_robot_box(xyz, arm_links_info, self.rm_robot)
                xyz = xyz[mask]
                rgb = rgb[mask]
                gt_sem = gt_sem[mask]

            if self.rm_pc_outliers:
                xyz, rgb, point_idxs = self._rm_pc_outliers(xyz, rgb=rgb, return_idxs=True)
                gt_sem = gt_sem[point_idxs]

            # sampling points
            if len(xyz) > self.num_points:
                point_idxs = np.random.permutation(len(xyz))[:self.num_points]
            else:
                if self.same_npoints_per_example:
                    point_idxs = np.random.choice(xyz.shape[0], self.num_points, replace=True)
                else:
                    max_npoints = int(len(xyz) * np.random.uniform(0.95, 1))
                    point_idxs = np.random.permutation(len(xyz))[:max_npoints]

            xyz = xyz[point_idxs]
            rgb = rgb[point_idxs]
            gt_sem = gt_sem[point_idxs]
            height = xyz[:, -1] - self.TABLE_HEIGHT

            # robot_mask = self._get_mask_with_label_ids(gt_sem, robot_label_ids)
            robot_box = RobotBox(arm_links_info, keep_gripper=False)
            robot_point_idxs = robot_box.get_pc_overlap_ratio(xyz=xyz, return_indices=True)[1]
            robot_point_idxs = np.array(list(robot_point_idxs))
            robot_mask = np.zeros((xyz.shape[0], ), dtype=bool)
            if len(robot_point_idxs) > 0:
                robot_mask[robot_point_idxs] = True

            pc_label = np.zeros((gt_sem.shape[0], ), dtype=np.int32)
            pc_label[robot_mask] = 1
            for oname in ['object', 'target']:
                if oname in gt_act_obj_labels[keystep]:
                    v = gt_act_obj_labels[keystep][oname]
                    if self.pc_label_type != 'mix':
                        obj_label_ids = v[self.pc_label_type]
                    else:
                        obj_label_ids = v[random.choice(['coarse', 'fine'])]
                    obj_mask = self._get_mask_with_label_ids(gt_sem, obj_label_ids)
                    if 'zrange' in v:
                        obj_mask = obj_mask & (xyz[:, 2] > v['zrange'][0]) & (xyz[:, 2] < v['zrange'][1])
                    if self.pc_label_augment > 0: # only keep part of the gt labels
                        rand_idxs = np.arange(obj_mask.shape[0])[obj_mask]
                        rm_num = int(np.random.uniform(low=0, high=self.pc_label_augment) * len(rand_idxs))
                        rand_idxs = np.random.permutation(rand_idxs)[:rm_num]
                        obj_mask[rand_idxs] = False
                    if oname == 'object':
                        pc_label[obj_mask] = 2
                    else:   # target
                        pc_label[obj_mask] = 3

            # point cloud augmentation
            if self.augment_pc:
                xyz, ee_pose, gt_trajs = self._augment_pc(xyz, ee_pose, gt_trajs, self.aug_max_rot)
            gt_rots = np.stack(
                [self._get_rotation_from_quat(gt_action[3:-1]) for gt_action in gt_trajs], 0
            )

            # normalize point cloud
            if self.xyz_shift == 'none':
                centroid = np.zeros((3, ))
            elif self.xyz_shift == 'center':
                centroid = np.mean(xyz, 0)
            elif self.xyz_shift == 'gripper':
                centroid = copy.deepcopy(ee_pose[:3])
            if self.xyz_norm:
                radius = np.max(np.sqrt(np.sum((xyz - centroid) ** 2, axis=1)))
            else:
                radius = 1

            xyz = (xyz - centroid) / radius
            height = height / radius
            gt_trajs[:, :3] = (gt_trajs[:, :3] - centroid) / radius
            ee_pose[:3] = (ee_pose[:3] - centroid) / radius
            outs['pc_centroids'].append(centroid)
            outs['pc_radius'].append(radius)

            gt_trajs = np.concatenate([gt_trajs[:, :3], gt_rots, gt_trajs[:, -1:]], -1)

            pc_ft = xyz
            if self.use_height:
                pc_ft = np.concatenate([pc_ft, height[:, None]], 1)
            if self.use_color:
                rgb = (rgb / 255.) * 2 - 1
                pc_ft = np.concatenate([pc_ft, rgb], 1)

            if self.pos_type == 'disc':
                gt_trajs_disc_pos_probs = []
                for gt_action in gt_trajs:
                    # (npoints, 3, pos_bins*2)
                    disc_pos_prob = get_disc_gt_pos_prob(
                        xyz, gt_action[:3], pos_bins=self.pos_bins, 
                        pos_bin_size=self.pos_bin_size,
                        heatmap_type=self.pos_heatmap_type,
                        robot_point_idxs=robot_point_idxs if self.pos_heatmap_no_robot else None
                    )
                    gt_trajs_disc_pos_probs.append(disc_pos_prob)
                # (max_trajs, npoints, 3, pos_bins*2)
                gt_trajs_disc_pos_probs = np.stack(gt_trajs_disc_pos_probs, 0)
                outs['gt_trajs_disc_pos_probs'].append(torch.from_numpy(gt_trajs_disc_pos_probs))
            
            outs['data_ids'].append(f'{taskvar}-{data_id.decode("ascii")}-t{t}')
            outs['pc_fts'].append(torch.from_numpy(pc_ft).float())
            outs['pc_centroids'].append(centroid)
            outs['pc_radius'].append(radius)
            outs['pc_labels'].append(torch.from_numpy(pc_label).long())
            outs['txt_embeds'].append(torch.from_numpy(action_embed).float())
            outs['ee_poses'].append(torch.from_numpy(ee_pose).float())
            outs['gt_trajs'].append(torch.from_numpy(gt_trajs).float())
            outs['gt_trajs_stop'].append(torch.arange(self.max_traj_len) >= (gt_traj_len-1))
            
        return outs
    

def base_collate_fn_partial(max_traj_len, data):
    batch = {}
    for key in data[0].keys():
        batch[key] = sum([x[key] for x in data], [])
    
    for key in ['pc_fts', 'pc_labels', 'ee_poses']:   
        batch[key] = torch.stack(batch[key], 0)

    gt_trajs, traj_lens = [], []
    for traj in batch['gt_trajs']:
        traj_lens.append(traj.size(0))
        if traj.size(0) < max_traj_len:
            # repeat the last action
            gt_trajs.append(
                torch.cat([traj, traj[-1].repeat(max_traj_len - traj.size(0), 1)])
            )
        else:
            assert len(traj) == max_traj_len, len(traj)
            gt_trajs.append(traj)
    batch['gt_trajs'] = torch.stack(gt_trajs, 0)
    batch['traj_masks'] = torch.from_numpy(
        gen_seq_masks(traj_lens, max_len=max_traj_len)
    ).bool()

    txt_lens = [x.size(0) for x in batch['txt_embeds']]
    batch['txt_masks'] = torch.from_numpy(
        gen_seq_masks(txt_lens, max_len=max(txt_lens))
    ).bool()
    batch['txt_embeds'] = pad_tensors(
        batch['txt_embeds'], lens=txt_lens, max_len=max(txt_lens)
    )

    if len(batch['pc_centroids']) > 0:
        batch['pc_centroids'] = np.stack(batch['pc_centroids'], 0)
        batch['pc_radius'] = np.array(batch['pc_radius'])
    
    return batch

def ptv3_collate_fn_partial(max_traj_len, data):
    batch = {}
    for key in data[0].keys():
        batch[key] = sum([x[key] for x in data], [])
    
    npoints_in_batch = [x.size(0) for x in batch['pc_fts']]
    batch['npoints_in_batch'] = npoints_in_batch
    batch['offset'] = torch.cumsum(torch.LongTensor(npoints_in_batch), dim=0)
    batch['pc_fts'] = torch.cat(batch['pc_fts'], 0)         # (#all points, 6)
    batch['pc_labels'] = torch.cat(batch['pc_labels'], 0)   # (#all points, )

    for key in ['ee_poses', 'gt_trajs_stop']:
        if key in batch:
            batch[key] = torch.stack(batch[key], 0)

    batch['txt_lens'] = [x.size(0) for x in batch['txt_embeds']]
    batch['txt_embeds'] = torch.cat(batch['txt_embeds'], 0)

    if len(batch['pc_centroids']) > 0:
        batch['pc_centroids'] = np.stack(batch['pc_centroids'], 0)

    gt_trajs, traj_lens = [], []
    for traj in batch['gt_trajs']:
        traj_lens.append(traj.size(0))
        if traj.size(0) < max_traj_len:
            # repeat the last action
            gt_trajs.append(
                torch.cat([traj, traj[-1].repeat(max_traj_len - traj.size(0), 1)])
            )
        else:
            assert len(traj) == max_traj_len, len(traj)
            gt_trajs.append(traj)
    batch['gt_trajs'] = torch.stack(gt_trajs, 0)    # (batch, traj_len, dim)
    batch['traj_lens'] = traj_lens
    batch['traj_masks'] = torch.from_numpy(
        gen_seq_masks(traj_lens, max_len=max_traj_len)
    ).bool()

    gt_trajs_disc_pos_probs = []
    for traj in batch['gt_trajs_disc_pos_probs']:
        if traj.size(0) < max_traj_len:
            # repeat the last action (max_traj_len, 3, npoints*bins)
            gt_trajs_disc_pos_probs.append(
                torch.cat([traj, traj[-1].repeat(max_traj_len - traj.size(0), 1, 1)])
            )
        else:
            assert len(traj) == max_traj_len, len(traj)
            gt_trajs_disc_pos_probs.append(traj)
    batch['gt_trajs_disc_pos_probs'] = gt_trajs_disc_pos_probs
    
    return batch


if __name__ == '__main__':
    from functools import partial

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    max_traj_len = 5

    # dataset = MotionPlannerDataset(
    #     'data/gembench/train_dataset/motion_keysteps_bbox_pcd/seed0/voxel1cm',
    #     'data/gembench/train_dataset/motion_keysteps_bbox_pcd/action_embeds_clip.npy',
    #     'assets/taskvars_train_target_label.json',
    #     taskvar_file='assets/taskvars_train.json', 
    #     num_points=4096, xyz_shift='none', xyz_norm=False, use_height=True,
    #     max_traj_len=max_traj_len, pc_label_type='coarse',
    #     pc_label_augment=False, pc_midstep_augment=True, augment_pc=True,
    #     rot_type='euler_disc', instr_embed_type='last', all_step_in_batch=True,
    #     rm_robot='box_keep_gripper', include_last_step=False, 
    #     same_npoints_per_example=False,
    #     rm_pc_outliers=False, rm_pc_outliers_neighbors=25, euler_resolution=5,
    #     pos_type='disc', pos_bins=15, pos_bin_size=0.01, pos_heatmap_type='dist',
    #     pos_heatmap_no_robot=True,
    # )
    dataset = MotionPlannerRealRobotDataset(
        'data/real_robot_data/v3/keysteps_bbox_pcd_cam2_motionplanner_vlm',
        'data/gembench/train_dataset/motion_keysteps_bbox_pcd/action_embeds_clip.npy',
        taskvar_file='assets/taskvars_realrobotv1.json', 
        num_points=4096, xyz_shift='none', xyz_norm=False, use_height=True,
        max_traj_len=max_traj_len, pc_label_type='coarse',
        pc_label_augment=False, pc_midstep_augment=True, augment_pc=True,
        rot_type='euler_disc', instr_embed_type='last', all_step_in_batch=True,
        rm_robot='box_keep_gripper', include_last_step=False, 
        same_npoints_per_example=False,
        rm_pc_outliers=False, rm_pc_outliers_neighbors=25, euler_resolution=5,
        pos_type='disc', pos_bins=15, pos_bin_size=0.01, pos_heatmap_type='dist',
        pos_heatmap_no_robot=True, real_robot=True,
    )
    print('#data', len(dataset))

    collate_fn = partial(ptv3_collate_fn_partial, max_traj_len)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=0, 
        collate_fn=collate_fn
    )
    print('#steps', len(dataloader))

    for batch in dataloader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(k, v.size())
            else:
                print(k)
        # print(batch['gt_trajs'])
        # print(batch['traj_masks'])
        np.save('batch.npy', batch)
        break
