import os
import numpy as np
import json
import copy
import random
from scipy.special import softmax

import lmdb
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

import torch
from torch.utils.data import Dataset

# import open3d as o3d
from sklearn.neighbors import LocalOutlierFactor
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


class SimplePolicyDataset(Dataset):
    def __init__(
            self, data_dir, instr_embed_file, taskvar_instr_file, taskvar_file=None,
            num_points=10000, xyz_shift='center', xyz_norm=True, use_height=False,
            rot_type='quat', instr_embed_type='last', all_step_in_batch=True, 
            rm_table=True, rm_robot='none', include_last_step=False, augment_pc=False,
            sample_points_by_distance=False, same_npoints_per_example=False, 
            rm_pc_outliers=False, rm_pc_outliers_neighbors=25, euler_resolution=5,
            pos_type='cont', pos_bins=50, pos_bin_size=0.01, 
            pos_heatmap_type='plain', pos_heatmap_no_robot=False,
            aug_max_rot=45, real_robot=False, **kwargs
        ):

        assert instr_embed_type in ['last', 'all']
        assert xyz_shift in ['none', 'center', 'gripper']
        assert pos_type in ['cont', 'disc']
        assert rot_type in ['quat', 'rot6d', 'euler', 'euler_delta', 'euler_disc']
        assert rm_robot in ['none', 'gt', 'box', 'box_keep_gripper']

        self.taskvar_instrs = json.load(open(taskvar_instr_file))
        self.instr_embeds = np.load(instr_embed_file, allow_pickle=True).item()
        if instr_embed_type == 'last':
            self.instr_embeds = {instr: embeds[-1:] for instr, embeds in self.instr_embeds.items()}
            
        if taskvar_file is not None:
            self.taskvars = json.load(open(taskvar_file))
        else:
            self.taskvars = os.listdir(data_dir)

        self.lmdb_envs, self.lmdb_txns = {}, {}
        self.data_ids = []
        for taskvar in self.taskvars:
            if not os.path.exists(os.path.join(data_dir, taskvar)):
                continue
            self.lmdb_envs[taskvar] = lmdb.open(os.path.join(data_dir, taskvar), readonly=True, lock=False)
            self.lmdb_txns[taskvar] = self.lmdb_envs[taskvar].begin()
            if all_step_in_batch:
                self.data_ids.extend(
                    [(taskvar, key) for key in self.lmdb_txns[taskvar].cursor().iternext(values=False)]
                )
            else:
                for key, value in self.lmdb_txns[taskvar].cursor():
                    value = msgpack.unpackb(value)
                    if include_last_step:
                        self.data_ids.extend([(taskvar, key, t) for t in range(len(value['xyz']))])
                    else:
                        self.data_ids.extend([(taskvar, key, t) for t in range(len(value['xyz']) - 1)])

        self.num_points = num_points
        self.xyz_shift = xyz_shift
        self.xyz_norm = xyz_norm
        self.use_height = use_height
        self.pos_type = pos_type
        self.rot_type = rot_type
        self.rm_table = rm_table
        self.rm_robot = rm_robot
        self.all_step_in_batch = all_step_in_batch
        self.include_last_step = include_last_step
        self.augment_pc = augment_pc
        self.aug_max_rot = np.deg2rad(aug_max_rot)
        self.sample_points_by_distance = sample_points_by_distance
        self.rm_pc_outliers = rm_pc_outliers
        self.rm_pc_outliers_neighbors = rm_pc_outliers_neighbors
        self.same_npoints_per_example = same_npoints_per_example
        self.euler_resolution = euler_resolution
        self.pos_bins = pos_bins
        self.pos_bin_size = pos_bin_size
        self.pos_heatmap_type = pos_heatmap_type
        self.pos_heatmap_no_robot = pos_heatmap_no_robot
        self.real_robot = real_robot

        self.TABLE_HEIGHT = get_robot_workspace(real_robot=real_robot)['TABLE_HEIGHT']
        self.rotation_transform = RotationMatrixTransform()

    def __exit__(self):
        for lmdb_env in self.lmdb_envs.values():
            lmdb_env.close()

    def __len__(self):
        return len(self.data_ids)

    def _get_mask_with_label_ids(self, sem, label_ids):
        mask = sem == label_ids[0]
        for label_id in label_ids[1:]:
            mask = mask | (sem == label_id)
        return mask
    
    def _get_mask_with_robot_box(self, xyz, arm_links_info, rm_robot_type):
        if rm_robot_type == 'box_keep_gripper':
            keep_gripper = True
        else:
            keep_gripper = False
        robot_box = RobotBox(
            arm_links_info, keep_gripper=keep_gripper, 
            env_name='real' if self.real_robot else 'rlbench'
        )
        _, robot_point_ids = robot_box.get_pc_overlap_ratio(xyz=xyz, return_indices=True)
        robot_point_ids = np.array(list(robot_point_ids))
        mask = np.ones((xyz.shape[0], ), dtype=bool)
        if len(robot_point_ids) > 0:
            mask[robot_point_ids] = False
        return mask
    
    def _rm_pc_outliers(self, xyz, rgb=None, return_idxs=False):
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(xyz)
        # pcd, idxs = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
        # pcd, idxs = pcd.remove_radius_outlier(nb_points=16, radius=0.03)
        clf = LocalOutlierFactor(n_neighbors=self.rm_pc_outliers_neighbors)
        preds = clf.fit_predict(xyz)
        idxs = (preds == 1)
        xyz = xyz[idxs]
        if rgb is not None:
            rgb = rgb[idxs]
        if return_idxs:
            return xyz, rgb, idxs
        else:
            return xyz, rgb
    
    def _rotate_gripper(self, gripper_rot, angle):
        rot = R.from_euler('z', angle, degrees=False)
        gripper_rot = R.from_quat(gripper_rot)
        gripper_rot = (rot * gripper_rot).as_quat()
        return gripper_rot
    
    def _augment_pc(self, xyz, ee_pose, gt_action, gt_rot, aug_max_rot):
        # rotate around z-axis
        angle = np.random.uniform(-1, 1) * aug_max_rot
        xyz = random_rotate_z(xyz, angle=angle)
        ee_pose[:3] = random_rotate_z(ee_pose[:3], angle=angle)
        gt_action[:3] = random_rotate_z(gt_action[:3], angle=angle)
        ee_pose[3:-1] = self._rotate_gripper(ee_pose[3:-1], angle)
        gt_action[3:-1] = self._rotate_gripper(gt_action[3:-1], angle)
        if self.rot_type == 'quat':
            gt_rot = gt_action[3:-1]
        elif self.rot_type == 'euler':
            gt_rot = self.rotation_transform.quaternion_to_euler(
                torch.from_numpy(gt_action[3:-1][None, :]))[0].numpy() / 180.
        elif self.rot_type == 'euler_disc':
            gt_rot = quaternion_to_discrete_euler(gt_action[3:-1], self.euler_resolution)
        elif self.rot_type == 'rot6d':
            gt_rot = self.rotation_transform.quaternion_to_ortho6d(
                torch.from_numpy(gt_action[3:-1][None, :]))[0].numpy()

        # add small noises (+-2mm)
        pc_noises = np.random.uniform(0, 0.002, size=xyz.shape)
        xyz = pc_noises + xyz

        return xyz, ee_pose, gt_action, gt_rot
    
    def get_groundtruth_rotations(self, ee_poses):
        gt_rots = torch.from_numpy(ee_poses)   # quaternions
        if self.rot_type == 'euler':    # [-1, 1]
            gt_rots = self.rotation_transform.quaternion_to_euler(gt_rots[1:]) / 180.
            gt_rots = torch.cat([gt_rots, gt_rots[-1:]], 0)
        elif self.rot_type == 'euler_disc': # 3D
            gt_rots = [quaternion_to_discrete_euler(x, self.euler_resolution) for x in gt_rots[1:]]
            gt_rots = torch.from_numpy(np.stack(gt_rots + gt_rots[-1:]))
        elif self.rot_type == 'euler_delta':
            gt_eulers = self.rotation_transform.quaternion_to_euler(gt_rots)
            gt_rots = (gt_eulers[1:] - gt_eulers[:-1]) % 360
            gt_rots[gt_rots > 180] -= 360
            gt_rots = gt_rots / 180.
            gt_rots = torch.cat([gt_rots, torch.zeros(1, 3)], 0)
        elif self.rot_type == 'rot6d':
            gt_rots = self.rotation_transform.quaternion_to_ortho6d(gt_rots)
            gt_rots = torch.cat([gt_rots, gt_rots[-1:]], 0)
        else:
            gt_rots = torch.cat([gt_rots, gt_rots[-1:]], 0)
        gt_rots = gt_rots.numpy()
        return gt_rots
    
    def __getitem__(self, idx):
        if self.all_step_in_batch:
            taskvar, data_id = self.data_ids[idx]
        else:
            taskvar, data_id, data_step = self.data_ids[idx]

        task, variation = taskvar.split('+')

        data = msgpack.unpackb(self.lmdb_txns[taskvar].get(data_id))

        outs = {
            'data_ids': [], 'pc_fts': [], 'step_ids': [],
            'pc_centroids': [], 'pc_radius': [], 'ee_poses': [], 
            'txt_embeds': [], 'gt_actions': [],
        }
        if self.pos_type == 'disc':
            outs['disc_pos_probs'] = []

        gt_rots = self.get_groundtruth_rotations(data['action'][:, 3:7])

        num_steps = len(data['xyz'])
        for t in range(num_steps):
            if (not self.all_step_in_batch) and t != data_step:
                continue
            if (not self.include_last_step) and (t == (num_steps - 1)):
                # the last step is the end observation
                continue

            xyz, rgb = data['xyz'][t], data['rgb'][t]
            # # real robot point cloud is very noisy, requiring noise point cloud removal
            # # segmentation fault if n_workers>0
            # if self.real_robot:
            #     pcd = o3d.geometry.PointCloud()
            #     pcd.points = o3d.utility.Vector3dVector(xyz)
            #     pcd.colors = o3d.utility.Vector3dVector(rgb)
            #     pcd, outlier_masks = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.2)
            #     xyz = xyz[outlier_masks]
            #     rgb = rgb[outlier_masks]

            if self.real_robot: # save in a different format
                arm_links_info = (data['bbox_info'][0], data['pose_info'][0])
            else:
                arm_links_info = (
                    {k: v[t] for k, v in data['bbox_info'].items()}, 
                    {k: v[t] for k, v in data['pose_info'].items()}
                )
            
            if t < num_steps - 1:
                gt_action = copy.deepcopy(data['action'][t+1])
            else:
                gt_action = copy.deepcopy(data['action'][-1])
            ee_pose = copy.deepcopy(data['action'][t])
            gt_rot = gt_rots[t]

            # randomly select one instruction
            instr = random.choice(self.taskvar_instrs[taskvar])
            instr_embed = self.instr_embeds[instr]

            # remove background points (table, robot arm)
            if self.rm_table:
                mask = xyz[..., 2] > self.TABLE_HEIGHT
                xyz = xyz[mask]
                rgb = rgb[mask]
            if self.rm_robot.startswith('box'):
                mask = self._get_mask_with_robot_box(xyz, arm_links_info, self.rm_robot)
                xyz = xyz[mask]
                rgb = rgb[mask]

            # TODO: segmentation fault in cleps with num_workers>0
            if self.rm_pc_outliers:
                xyz, rgb = self._rm_pc_outliers(xyz, rgb)

            # sampling points
            if len(xyz) > self.num_points:
                if self.sample_points_by_distance:
                    dists = np.sqrt(np.sum((xyz - ee_pose[:3])**2, 1))
                    probs = 1 / np.maximum(dists, 0.1)
                    probs = np.maximum(softmax(probs), 1e-30) 
                    probs = probs / sum(probs)
                    # probs = 1 / dists
                    # probs = probs / np.sum(probs)
                    point_idxs = np.random.choice(len(xyz), self.num_points, replace=False, p=probs)
                else:
                    point_idxs = np.random.choice(len(xyz), self.num_points, replace=False)
            else:
                if self.same_npoints_per_example:
                    point_idxs = np.random.choice(xyz.shape[0], self.num_points, replace=True)
                else:
                    max_npoints = int(len(xyz) * np.random.uniform(0.95, 1))
                    point_idxs = np.random.permutation(len(xyz))[:max_npoints]

            xyz = xyz[point_idxs]
            rgb = rgb[point_idxs]
            height = xyz[:, -1] - self.TABLE_HEIGHT

            if self.pos_heatmap_no_robot:
                robot_box = RobotBox(
                    arm_links_info=arm_links_info,
                    env_name='real' if self.real_robot else 'rlbench'
                )
                robot_point_idxs = np.array(
                    list(robot_box.get_pc_overlap_ratio(xyz=xyz, return_indices=True)[1])
                )
            else:
                robot_point_idxs = None

            # point cloud augmentation
            if self.augment_pc:
                xyz, ee_pose, gt_action, gt_rot = self._augment_pc(
                    xyz, ee_pose, gt_action, gt_rot, self.aug_max_rot
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
            gt_action[:3] = (gt_action[:3] - centroid) / radius
            ee_pose[:3] = (ee_pose[:3] - centroid) / radius
            outs['pc_centroids'].append(centroid)
            outs['pc_radius'].append(radius)

            gt_action = np.concatenate([gt_action[:3], gt_rot, gt_action[-1:]], 0)

            rgb = (rgb / 255.) * 2 - 1
            pc_ft = np.concatenate([xyz, rgb], 1)
            if self.use_height:
                pc_ft = np.concatenate([pc_ft, height[:, None]], 1)

            if len(pc_ft) == 0:
                continue

            if self.pos_type == 'disc':
                # (npoints, 3, 100)
                disc_pos_prob = get_disc_gt_pos_prob(
                    xyz, gt_action[:3], pos_bins=self.pos_bins, 
                    pos_bin_size=self.pos_bin_size,
                    heatmap_type=self.pos_heatmap_type,
                    robot_point_idxs=robot_point_idxs
                )
                outs['disc_pos_probs'].append(torch.from_numpy(disc_pos_prob))
            
            outs['data_ids'].append(f'{taskvar}-{data_id.decode("ascii")}-t{t}')
            outs['pc_fts'].append(torch.from_numpy(pc_ft).float())
            outs['txt_embeds'].append(torch.from_numpy(instr_embed).float())
            outs['ee_poses'].append(torch.from_numpy(ee_pose).float())
            outs['gt_actions'].append(torch.from_numpy(gt_action).float())
            outs['step_ids'].append(t)

        return outs
    


def base_collate_fn(data):
    batch = {}
    for key in data[0].keys():
        batch[key] = sum([x[key] for x in data], [])
    
    for key in ['pc_fts', 'ee_poses', 'gt_actions']:   
        batch[key] = torch.stack(batch[key], 0)

    batch['step_ids'] = torch.LongTensor(batch['step_ids'])

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

def ptv3_collate_fn(data):
    batch = {}
    for key in data[0].keys():
        batch[key] = sum([x[key] for x in data], [])
    
    npoints_in_batch = [x.size(0) for x in batch['pc_fts']]
    batch['npoints_in_batch'] = npoints_in_batch
    batch['offset'] = torch.cumsum(torch.LongTensor(npoints_in_batch), dim=0)
    batch['pc_fts'] = torch.cat(batch['pc_fts'], 0) # (#all points, 6)

    for key in ['ee_poses', 'gt_actions']:   
        batch[key] = torch.stack(batch[key], 0)

    # if 'disc_pos_probs' in batch:
    #     batch['disc_pos_probs'] = batch['disc_pos_probs'] # [(3, #all pointspos_bins*2)]

    batch['step_ids'] = torch.LongTensor(batch['step_ids'])

    batch['txt_lens'] = [x.size(0) for x in batch['txt_embeds']]
    batch['txt_embeds'] = torch.cat(batch['txt_embeds'], 0)

    if len(batch['pc_centroids']) > 0:
        batch['pc_centroids'] = np.stack(batch['pc_centroids'], 0)

    return batch


if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dataset = SimplePolicyDataset(
        'data/gembench/train_dataset/keysteps_bbox_pcd/seed0/voxel1cm',
        'data/gembench/train_dataset/keysteps_bbox_pcd/instr_embeds_clip.npy',
        'assets/taskvars_instructions_new.json',
        taskvar_file='assets/taskvars_train.json', 
        num_points=4096, xyz_norm=True, xyz_shift='center',
        use_height=False, rot_type='euler_delta', 
        instr_embed_type='last', include_last_step=True,
        rm_robot='box_keep_gripper', rm_table=True,
        all_step_in_batch=True, same_npoints_per_example=False,
        sample_points_by_distance=True, augment_pc=False,
        rm_pc_outliers=True
    )
    print('#data', len(dataset))

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=True, num_workers=0, 
        collate_fn=ptv3_collate_fn
    )
    print('#steps', len(dataloader))

    for batch in dataloader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(k, v.size())
            # else:
            #     print(k, v)
        print(batch['gt_actions'].min(dim=0)[0])
        print(batch['gt_actions'].max(dim=0)[0])
        print(np.min(batch['pc_radius']), np.mean(batch['pc_radius']), np.max(batch['pc_radius']))
        # np.save('batch.npy', batch)
        break
