import os
import random
import json
import re
import numpy as np
from easydict import EasyDict
import copy

import torch

import json
import copy
import numpy as np
import torch

from genrobo3d.configs.rlbench.constants import get_robot_workspace
from genrobo3d.utils.point_cloud import voxelize_pcd, get_pc_foreground_mask
from genrobo3d.utils.robot_box import RobotBox

from genrobo3d.vlm_models.clip_encoder import ClipEncoder
from genrobo3d.models.motion_planner_ptv3 import (
    MotionPlannerPTV3AdaNorm, MotionPlannerPTV3CA
)
from genrobo3d.configs.default import get_config as get_model_config
from genrobo3d.evaluation.common import load_checkpoint, parse_code


class GroundtruthTaskPlanner(object):
    def __init__(self, gt_plan_file):
        self.taskvar_plans = {}
        with open(gt_plan_file) as f:
            for x in f:
                x = x.strip()
                if len(x) == 0:
                    continue
                if x.startswith('# taskvar: '):
                    taskvar = x.split('# taskvar: ')[-1]
                    self.taskvar_plans[taskvar] = []
                elif not x.startswith('#'):
                    self.taskvar_plans[taskvar].append(x)

    def __call__(self, taskvar):
        plans = self.taskvar_plans[taskvar]
        return plans

    def estimate_height_range(self, target_name, obj_height):
        if 'middle bottom' in target_name:
            zrange = [obj_height/4*1, obj_height/4*2]
        elif 'middle top' in target_name:
            zrange = [obj_height/4*2, obj_height/4*3]
        elif 'bottom' in target_name:
            zrange = [0, obj_height/3]
        elif 'middle' in target_name:
            zrange = [obj_height/3, obj_height/3*2]
        elif 'top' in target_name:
            zrange = [obj_height/3*2, obj_height]
        else:
            zrange = [0, obj_height]
        return np.array(zrange)

class GroundtruthVision(object):
    def __init__(
        self, gt_label_file, num_points=4096, voxel_size=0.01, 
        same_npoints_per_example=False, rm_robot='box_keep_gripper',
        xyz_shift='center', xyz_norm=False, use_height=True,
        pc_label_type='coarse', use_color=False,
    ):
        self.taskvar_gt_target_labels = json.load(open(gt_label_file))
        self.workspace = get_robot_workspace(real_robot=False)
        self.TABLE_HEIGHT = self.workspace['TABLE_HEIGHT']

        self.num_points = num_points
        self.voxel_size = voxel_size
        self.pc_label_type = pc_label_type
        self.same_npoints_per_example = same_npoints_per_example
        self.rm_robot = rm_robot
        self.xyz_shift = xyz_shift
        self.xyz_norm = xyz_norm
        self.use_height = use_height
        self.use_color = use_color

    def __call__(
        self, taskvar, step_id, pcd_images, sem_images, gripper_pose, arm_links_info, 
        rgb_images=None,
    ):
        task, variation = taskvar.split('+')
        pcd_xyz = pcd_images.reshape(-1, 3)
        pcd_sem = sem_images.reshape(-1)
        if self.use_color:
            assert rgb_images is not None
            pcd_rgb = rgb_images.reshape(-1, 3)

        # remove background and table points
        fg_mask = get_pc_foreground_mask(pcd_xyz, self.workspace)
        pcd_xyz = pcd_xyz[fg_mask]
        pcd_sem = pcd_sem[fg_mask]
        if self.use_color:
            pcd_rgb = pcd_rgb[fg_mask]

        pcd_xyz, idxs = voxelize_pcd(pcd_xyz, voxel_size=self.voxel_size)
        pcd_sem = pcd_sem[idxs]
        if self.use_color:
            pcd_rgb = pcd_rgb[idxs]

        if self.rm_robot != 'none':
            if self.rm_robot == 'box':
                robot_box = RobotBox(arm_links_info, keep_gripper=False)
            elif self.rm_robot == 'box_keep_gripper':
                robot_box = RobotBox(arm_links_info, keep_gripper=True)
            robot_point_idxs = robot_box.get_pc_overlap_ratio(xyz=pcd_xyz, return_indices=True)[1]
            robot_point_idxs = np.array(list(robot_point_idxs))
            if len(robot_point_idxs) > 0:
                mask = np.ones((pcd_xyz.shape[0], ), dtype=bool)
                mask[robot_point_idxs] = False
                pcd_xyz = pcd_xyz[mask]
                pcd_sem = pcd_sem[mask]
                if self.use_color:
                    pcd_rgb = pcd_rgb[mask]

        # sample points
        if len(pcd_xyz) > self.num_points:
            point_idxs = np.random.permutation(len(pcd_xyz))[:self.num_points]
        else:
            if self.same_npoints_per_example:
                point_idxs = np.random.choice(pcd_xyz.shape[0], self.num_points, replace=True)
            else:
                point_idxs = np.arange(pcd_xyz.shape[0])
        pcd_xyz = pcd_xyz[point_idxs]
        pcd_sem = pcd_sem[point_idxs]
        height = pcd_xyz[..., 2] - self.TABLE_HEIGHT
        if self.use_color:
            pcd_rgb = pcd_rgb[point_idxs]

        # robot pcd_label
        pcd_label = np.zeros_like(pcd_sem)
        robot_box = RobotBox(arm_links_info, keep_gripper=False)
        robot_point_idxs = robot_box.get_pc_overlap_ratio(xyz=pcd_xyz, return_indices=True)[1]
        robot_point_idxs = np.array(list(robot_point_idxs))
        if len(robot_point_idxs) > 0:
            pcd_label[robot_point_idxs] = 1
        for query_key, query_label_id in zip(['object', 'target'], [2, 3]):
            if query_key in self.taskvar_gt_target_labels[taskvar][step_id]:
                gt_target_labels = self.taskvar_gt_target_labels[taskvar][step_id][query_key]
                gt_query_mask = [pcd_sem == x for x in gt_target_labels[self.pc_label_type]]
                gt_query_mask = np.sum(gt_query_mask, 0) > 0
                if 'zrange' in gt_target_labels:
                    gt_query_mask = gt_query_mask & (pcd_xyz[..., 2] > gt_target_labels['zrange'][0]) & (pcd_xyz[..., 2] < gt_target_labels['zrange'][1])
                pcd_label[gt_query_mask] = query_label_id

        # normalize point cloud
        if self.xyz_shift == 'none':
            pc_centroid = np.zeros((3, ))
        elif self.xyz_shift == 'center':
            pc_centroid = np.mean(pcd_xyz, 0)
        elif self.xyz_shift == 'gripper':
            pc_centroid = copy.deepcopy(gripper_pose[:3])
        if self.xyz_norm:
            pc_radius = np.max(np.sqrt(np.sum((pcd_xyz - pc_centroid) ** 2, axis=1)))
        else:
            pc_radius = 1
        pcd_xyz = (pcd_xyz - pc_centroid) / pc_radius
        gripper_pose[:3] = (gripper_pose[:3] - pc_centroid) / pc_radius
        
        pcd_ft = pcd_xyz
        if self.use_height:
            pcd_ft = np.concatenate([pcd_ft, height[:, None]], -1)
        if self.use_color:
            pcd_rgb = (pcd_rgb / 255.) * 2 - 1
            pcd_ft = np.concatenate([pcd_ft, pcd_rgb], -1)

        outs = {
            'pc_fts': torch.from_numpy(pcd_ft).float(),
            'pc_labels': torch.from_numpy(pcd_label).long(),
            'offset': torch.LongTensor([pcd_xyz.shape[0]]),
            'npoints_in_batch': [pcd_xyz.shape[0]],
            'pc_centroids': pc_centroid,
            'pc_radius': pc_radius,
            'ee_poses': torch.from_numpy(gripper_pose).float().unsqueeze(0),
        }

        return outs


class GroundtruthRobotPipeline(object):
    def __init__(self, config) -> None:
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # build LLM high-level planner
        llm_config = config.llm_planner
        self.llm_planner = GroundtruthTaskPlanner(llm_config.gt_plan_file)
        
        mp_expr_dir = config.motion_planner.expr_dir
        mp_config_file = config.motion_planner.config_file
        mp_config = get_model_config(mp_config_file)
        data_cfg = mp_config.TRAIN_DATASET
        self.instr_include_objects = data_cfg.get('instr_include_objects', False)
        self.vlm_pipeline = GroundtruthVision(
            self.config.object_grounding.gt_label_file,
            num_points=data_cfg.num_points, voxel_size=mp_config.MODEL.action_config.voxel_size, 
            same_npoints_per_example=data_cfg.same_npoints_per_example, rm_robot=data_cfg.rm_robot,
            xyz_shift=data_cfg.xyz_shift, xyz_norm=data_cfg.xyz_norm, use_height=data_cfg.use_height,
            pc_label_type=data_cfg.pc_label_type if config.motion_planner.pc_label_type is None else config.motion_planner.pc_label_type, use_color=data_cfg.get('use_color', False),
        )

        # build motion planner
        # self.clip_model = OpenClipEncoder(device=self.device) # to encode action/object texts
        self.clip_model = ClipEncoder(device=self.device)
        self.motion_planner = self.build_motion_planner(config.motion_planner, device=self.device)        

        # caches
        self.set_system_caches()

    def set_system_caches(self):
        self.action_embeds, self.query_embeds = {}, {}

    def build_motion_planner(self, mp_config, device):
        mp_model_config = get_model_config(mp_config.config_file)
        if mp_model_config.MODEL.model_class == 'MotionPlannerPTV3CA':
            motion_planner = MotionPlannerPTV3CA(mp_model_config.MODEL).to(self.device)
        else:
            motion_planner = MotionPlannerPTV3AdaNorm(mp_model_config.MODEL).to(self.device)
        motion_planner.eval()
        load_checkpoint(motion_planner, mp_config.checkpoint)
        return motion_planner

    @torch.no_grad()
    def predict(self, task_str, variation, step_id, obs_state_dict, episode_id, instructions, cache=None):
        taskvar = f'{task_str}+{variation}'

        if step_id == 0:
            cache = EasyDict(
                valid_actions = [], object_vars = {}
            )
            if self.config.motion_planner.save_obs_outs:
                cache.episode_outdir = os.path.join(
                    self.config.motion_planner.pred_dir, 'obs_outs', 
                    f'{task_str}+{variation}', f'{episode_id}'
                )
                os.makedirs(cache.episode_outdir, exist_ok=True)
            else:
                cache.episode_outdir = None

        # print(f'taskvar={task_str}+{variation}, step={step_id}, #cache_actions={len(self.cache.valid_actions)}')

        if len(cache.valid_actions) > 0:
            cur_action = cache.valid_actions[0][:8]
            cache.valid_actions = cache.valid_actions[1:]
            out = {"action": cur_action, 'cache': cache}
            if cache.episode_outdir:
                np.save(
                    os.path.join(cache.episode_outdir, f'{step_id}.npy'),
                    {
                        'obs': obs_state_dict,
                        'action': cur_action
                    }
                )
            return out

        pcd_images = obs_state_dict['pc']
        rgb_images = obs_state_dict['rgb']
        sem_images = obs_state_dict['gt_mask']
        arm_links_info = obs_state_dict['arm_links_info']
        gripper_pose = copy.deepcopy(obs_state_dict['gripper'])

        # initialize: task planning
        if step_id == 0:
            if self.config.llm_planner.use_groundtruth:
                highlevel_plans = self.llm_planner(taskvar)
            
            # print('plans\n', highlevel_plans)
            cache.highlevel_plans = [parse_code(x) for x in highlevel_plans]
            cache.highlevel_step_id = 0
            cache.highlevel_step_id_norelease = 0
            # print('parsed plans\n', self.cache.highlevel_plans)

        # print('highlevel step id', self.cache.highlevel_step_id)
        if cache.highlevel_step_id >= len(cache.highlevel_plans):
            if self.config.pipeline.restart:
                cache.highlevel_step_id = 0
                cache.highlevel_step_id_norelease = 0
            else:
                return {'action': np.zeros((8, )), 'cache': cache}

        plan = cache.highlevel_plans[cache.highlevel_step_id]
        if plan is None:
            return {'action': np.zeros((8, ))}
        
        if plan['action'] == 'release':
            action = gripper_pose
            action[7] = 1
            cache.highlevel_step_id += 1
            return {'action': action, 'cache': cache}

        batch = self.vlm_pipeline(
            taskvar, cache.highlevel_step_id_norelease, 
            pcd_images, sem_images, gripper_pose, arm_links_info,
            rgb_images=rgb_images
        )

        # motion planning
        action_name = plan['action']
        if self.instr_include_objects:
            if 'object' in plan and plan['object'] is not None:
                object_name = ''.join([x for x in plan['object'] if not x.isdigit()])
                object_name = object_name.replace('_', ' ').strip()
                action_name = f"{action_name} {object_name}"
            if 'target' in plan and plan['target'] is not None and plan['target'] not in ['up', 'down', 'out', 'in']:
                # TODO: should keep the target name is target is a variable
                target_name = ''.join([x for x in plan['target'] if not x.isdigit()])
                target_name = target_name.replace('_', ' ').strip()
                action_name = f"{action_name} to {target_name}"
        # print(action_name)
        action_embeds = self.clip_model(
            'text', action_name, use_prompt=False, output_hidden_states=True
        )[0]    # shape=(txt_len, hidden_size)
        batch.update({
            'txt_embeds': action_embeds,
            'txt_lens':  [action_embeds.size(0)],
        })
        
        pred_actions = self.motion_planner(batch, compute_loss=False)[0] # (max_action_len, 8)
        pred_actions[:, 7:] = torch.sigmoid(pred_actions[:, 7:])
        pred_actions = pred_actions.data.cpu().numpy()
        # print(pred_actions)

        # rescale the predicted position
        pred_actions[:, :3] = (pred_actions[:, :3] * batch['pc_radius']) + batch['pc_centroids']
        pred_actions[:, 2] = np.maximum(pred_actions[:, 2], self.vlm_pipeline.TABLE_HEIGHT + 0.005)

        valid_actions = []
        for t, pred_action in enumerate(pred_actions):
            # if pred_action[8] >= 0.5 or self.config.pipeline.exceed_max_action:
            valid_actions.append(pred_action)
            if t + 1 >= self.config.motion_planner.run_action_step:
                break
            if pred_action[-1] > 0.5:
                break

        if pred_action[-1] > 0.5:
            cache.highlevel_step_id += 1
            cache.highlevel_step_id_norelease += 1
        
        cache.valid_actions = valid_actions[1:]
        # print('valid actions', len(valid_actions), valid_actions)
        out = {
            'action': valid_actions[0][:8],
            'cache': cache
        }

        if cache.episode_outdir is not None:
            del batch['txt_embeds'], batch['txt_lens']
            np.save(
                os.path.join(cache.episode_outdir, f'{step_id}.npy'),
                {
                    'batch': {k: v.data.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in batch.items()},
                    'obs': obs_state_dict,
                    'valid_actions': valid_actions,
                } # type: ignore
            )

        return out


