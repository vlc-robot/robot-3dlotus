import os
import random
import json
import numpy as np
from easydict import EasyDict
import copy
import collections
from scipy.spatial.transform import Rotation as R

import torch

from genrobo3d.configs.rlbench.constants import get_robot_workspace
from genrobo3d.utils.point_cloud import voxelize_pcd
from genrobo3d.evaluation.common import load_checkpoint, parse_code
from genrobo3d.configs.default import get_config as get_model_config

from genrobo3d.utils.robot_box import RobotBox
from genrobo3d.evaluation.robot_pipeline_gt import GroundtruthTaskPlanner
from genrobo3d.vlm_models.llm_task_planner import LlamaTaskPlanner
from genrobo3d.vlm_models.vlm_pipeline import VLMPipeline
from genrobo3d.vlm_models.clip_encoder import ClipEncoder
from genrobo3d.models.motion_planner_ptv3 import (
    MotionPlannerPTV3AdaNorm, MotionPlannerPTV3CA
)


class RobotPipeline(object):
    def __init__(self, config) -> None:
        
        self.config = config
        self.env_name = 'rlbench' if not config.pipeline.real_robot else 'real'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # build LLM high-level planner
        llm_config = config.llm_planner
        if llm_config.use_groundtruth:
            self.llm_planner = GroundtruthTaskPlanner(
                llm_config.gt_plan_file
            )
        else:
            self.llm_planner = LlamaTaskPlanner(
                llm_config.prompt_dir, llm_config.asset_dir,
                temperature=0, max_seq_len=8192, top_p=0.9, max_batch_size=1, 
                max_gen_len=None, device=None, master_port=llm_config.master_port,
                ckpt_dir=llm_config.ckpt_dir, groq_model=llm_config.groq_model,
                cache_file=llm_config.cache_file
            )
        
        # self.task_objects = json.load(open(config.env.task_object_file))

        # build VLM for coarse-grained object grounding
        self.vlm_pipeline = VLMPipeline(
            sam_model_id="huge", det_model_id="large",
            use_2d_caption=False, use_3d_caption=False,
            env_name=self.env_name
        )

        # build motion planner
        self.clip_model = ClipEncoder() # to encode action/object texts
        self.motion_planner, self.mp_config = self.build_motion_planner(config.motion_planner)        

        # caches
        self.set_system_caches()
        self.workspace = get_robot_workspace(
            real_robot=config.pipeline.real_robot, use_vlm=True,
        )

    def set_system_caches(self):
        self.action_embeds, self.query_embeds = {}, {}

    def build_motion_planner(self, mp_config):
        mp_model_config = get_model_config(mp_config.config_file)
        if mp_model_config.MODEL.model_class == 'MotionPlannerPTV3CA':
            motion_planner = MotionPlannerPTV3CA(mp_model_config.MODEL).to(self.device)
        else:
            motion_planner = MotionPlannerPTV3AdaNorm(mp_model_config.MODEL).to(self.device)
        motion_planner.eval()
        load_checkpoint(motion_planner, mp_config.checkpoint)
        return motion_planner, mp_model_config
    
    def prepare_motion_planner_input(
        self, vlm_results, plan, arm_links_info, gripper_pose, voxel_size=0.01, 
        rm_robot='none', num_points=4096, same_npoints_per_example=False,
        xyz_shift='center', xyz_norm=False, use_height=False, zrange=None,
        target_var_xyz=None, use_color=False, instr_include_objects=False,
    ):
        pcd_xyz, pcd_rgb, pcd_label = [], [], []
        for k, obj_data in enumerate(vlm_results.objects):
            pcd_xyz.append(obj_data.pcd_xyz)
            pcd_rgb.append(obj_data.pcd_rgb)
            pcd_label.append(np.zeros((obj_data.pcd_xyz.shape[0], ), dtype=np.int32))
            # print('obj', k, obj_data.captions, len(obj_data.pcd_xyz))
            if len(obj_data.captions) > 0 and obj_data.captions[0] == 'robot':
                pcd_label[-1][:] = 1

        mani_obj = EasyDict()
        for query_key in ['object', 'target']:
            if plan[query_key] is not None:
                query = plan[query_key]
                # coarse-grained object grounding
                best_obj_id, _, pred_sim = self.vlm_pipeline.ground_object_with_query(
                    query, objects=vlm_results.objects, debug=False, return_sims=True
                )
                if best_obj_id is not None:
                    # print(query, best_obj_id, pred_sim)
                    if query_key == 'object':
                        pcd_label[best_obj_id][:] = 2
                        mani_obj.pcd_xyz = pcd_xyz[best_obj_id]
                        mani_obj.name = plan['ret_val']
                        if zrange is not None:
                            pcd_label[best_obj_id][pcd_xyz[best_obj_id][:, 2] < zrange[0]] = 0
                            pcd_label[best_obj_id][pcd_xyz[best_obj_id][:, 2] > zrange[1]] = 0
                    elif query_key == 'target':
                        if target_var_xyz is not None:
                            target_var_xyz = torch.from_numpy(target_var_xyz).float().to(self.device).unsqueeze(0)
                            obj_target_var_dists = [self.vlm_pipeline.chamfer_dist_fn(
                                target_var_xyz, 
                                torch.from_numpy(obj.pcd_xyz).float().to(self.device).unsqueeze(0),
                                bidirectional=True
                                )[0].item()
                                for obj in vlm_results.objects if len(obj.captions) == 0]
                            best_obj_id = np.argmin(obj_target_var_dists)
                            # print('target var', obj_target_var_dists, best_obj_id)
                        pcd_label[best_obj_id][:] = 3
                        if zrange is not None:
                            pcd_label[best_obj_id][pcd_xyz[best_obj_id][:, 2] < zrange[0]] = 0
                            pcd_label[best_obj_id][pcd_xyz[best_obj_id][:, 2] > zrange[1]] = 0

        pcd_xyz = np.concatenate(pcd_xyz)
        pcd_rgb = np.concatenate(pcd_rgb)
        pcd_label = np.concatenate(pcd_label)
        # print('merged', pcd_xyz.shape, collections.Counter(pcd_label))
        
        # point cloud preprocessing
        pcd_xyz, idxs = voxelize_pcd(pcd_xyz, voxel_size=voxel_size)
        pcd_label = pcd_label[idxs]
        pcd_rgb = pcd_rgb[idxs]
        
        if rm_robot != 'none':
            if rm_robot == 'box':
                robot_box = RobotBox(arm_links_info, keep_gripper=False, env_name=self.env_name)
            elif rm_robot == 'box_keep_gripper':
                robot_box = RobotBox(arm_links_info, keep_gripper=True, env_name=self.env_name)
            robot_point_idxs = robot_box.get_pc_overlap_ratio(xyz=pcd_xyz, return_indices=True)[1]
            robot_point_idxs = np.array(list(robot_point_idxs))
            if len(robot_point_idxs) > 0:
                mask = np.ones((pcd_xyz.shape[0], ), dtype=bool)
                mask[robot_point_idxs] = False
                pcd_xyz = pcd_xyz[mask]
                pcd_label = pcd_label[mask]
                pcd_rgb = pcd_rgb[mask]
        
        # sample points
        if len(pcd_xyz) > num_points:
            point_idxs = np.random.permutation(len(pcd_xyz))[:num_points]
        else:
            if same_npoints_per_example:
                point_idxs = np.random.choice(pcd_xyz.shape[0], num_points, replace=True)
            else:
                point_idxs = np.arange(pcd_xyz.shape[0])
        pcd_xyz = pcd_xyz[point_idxs]
        pcd_label = pcd_label[point_idxs]
        pcd_height = pcd_xyz[:, 2] - self.workspace['TABLE_HEIGHT']
        pcd_rgb = pcd_rgb[point_idxs]

        # normalize point cloud
        if xyz_shift == 'none':
            pc_centroid = np.zeros((3, ))
        elif xyz_shift == 'center':
            pc_centroid = np.mean(pcd_xyz, 0)
        elif xyz_shift == 'gripper':
            pc_centroid = copy.deepcopy(gripper_pose[:3])
        if xyz_norm:
            pc_radius = np.max(np.sqrt(np.sum((pcd_xyz - pc_centroid) ** 2, axis=1)))
        else:
            pc_radius = 1
        pcd_xyz = (pcd_xyz - pc_centroid) / pc_radius
        gripper_pose[:3] = (gripper_pose[:3] - pc_centroid) / pc_radius

        pcd_ft = pcd_xyz
        if use_height:
            pcd_ft = np.concatenate([pcd_ft, pcd_height[:, None]], -1)
        if use_color:
            pcd_rgb = (pcd_rgb / 255.) * 2 - 1
            pcd_ft = np.concatenate([pcd_ft, pcd_rgb], -1)

        batch = {
            'pc_fts': torch.from_numpy(pcd_ft).float(),
            'pc_labels': torch.from_numpy(pcd_label).long(),
            'offset': torch.LongTensor([pcd_xyz.shape[0]]),
            'npoints_in_batch': [pcd_xyz.shape[0]],
            'pc_centroids': pc_centroid,
            'pc_radius': pc_radius,
            'ee_poses': torch.from_numpy(gripper_pose).float().unsqueeze(0),
        }

        action_name = plan['action']
        if plan['target'] in ['up', 'down', 'out', 'in']:
            action_name = action_name + ' ' + plan['target']
        if instr_include_objects:
            if 'object' in plan and plan['object'] is not None:
                object_name = ''.join([x for x in plan['object'] if not x.isdigit()])
                object_name = object_name.replace('_', ' ').strip()
                action_name = f"{action_name} {object_name}"
            if 'target' in plan and plan['target'] is not None and plan['target'] not in ['up', 'down', 'out', 'in']:
                target_name = ''.join([x for x in plan['target'] if not x.isdigit()])
                target_name = target_name.replace('_', ' ').strip()
                action_name = f"{action_name} to {target_name}"
        # print('action name', action_name)

        action_embeds = self.clip_model(
            'text', action_name, use_prompt=False, output_hidden_states=True
        )[0]    # shape=(txt_len, hidden_size)
        batch.update({
            'txt_embeds': action_embeds,
            'txt_lens':  [action_embeds.size(0)],
        })

        extra_outs = EasyDict()
        if len(mani_obj) > 0:
            extra_outs.mani_obj = mani_obj
        return batch, extra_outs
    
    def move_grasped_obj_xyz(self, cur_action, prev_pose, obj_xyz):
        # print('move object before', np.mean(obj_xyz, 0), cur_action, prev_pose)
        translation = cur_action[:3] - prev_pose[:3]
        rotation = R.from_quat(cur_action[3:7]).as_euler('xyz') - R.from_quat(prev_pose[3:7]).as_euler('xyz')
        rotation = R.from_euler('xyz', rotation)
        obj_xyz += translation
        obj_xyz = rotation.apply(obj_xyz)
        # print('move object', np.mean(obj_xyz, 0))
        return obj_xyz

    @torch.no_grad()
    def predict(self, task_str, variation, step_id, obs_state_dict, episode_id, instructions, cache=None):
        taskvar = f"{task_str}+{variation}"
        # print(f'taskvar={taskvar}, step={step_id}')

        if step_id == 0:
            cache = EasyDict(
                valid_actions = [], object_vars = {}, highlevel_plans = [],
                ret_objs = {}, grasped_obj_name = None, 
                prev_ee_pose = copy.deepcopy(obs_state_dict['gripper'])
            )
            if self.config.motion_planner.save_obs_outs:
                cache.episode_outdir = os.path.join(
                    self.config.motion_planner.pred_dir, 'obs_outs', taskvar, f'{episode_id}'
                )
                os.makedirs(cache.episode_outdir, exist_ok=True)
            else:
                cache.episode_outdir = None

        if len(cache.valid_actions) > 0:
            cur_action = cache.valid_actions[0][:8]
            cache.valid_actions = cache.valid_actions[1:]
            out = {"action": cur_action, 'cache': cache}
            plan = cache.highlevel_plans[cache.highlevel_step_id - 1]
            if cache.grasped_obj_name is not None and \
                cache.grasped_obj_name in cache.ret_objs and \
                plan['action'].startswith('move grasped object'):
                # rotate the grasped object position
                self.move_grasped_obj_xyz(out['action'], cache.prev_ee_pose, cache.ret_objs[cache.grasped_obj_name])
            cache.prev_ee_pose = out['action']
            # if cache.episode_outdir:
            #     np.save(
            #         os.path.join(cache.episode_outdir, f'{step_id}.npy'),
            #         {
            #             'obs': obs_state_dict,
            #             'action': cur_action
            #         }
            #     )
            return out

        rgb_images = obs_state_dict['rgb']
        pcd_images = obs_state_dict['pc']
        arm_links_info = obs_state_dict['arm_links_info']
        gripper_pose = copy.deepcopy(obs_state_dict['gripper'])

        # initialize: task planning
        if step_id == 0:
            # TODO: simply use the first one (might be better to use the longest one)
            instruction = instructions[0]
            if self.config.llm_planner.use_groundtruth:
                highlevel_plans = self.llm_planner(taskvar)
            else:
                _, highlevel_plans = self.llm_planner(
                    instruction, context=None, verbose=False
                )
            
            # print('plans\n', highlevel_plans)
            cache.highlevel_plans = [parse_code(x) for x in highlevel_plans]
            cache.highlevel_step_id = 0
            # print('parsed plans\n', self.cache.highlevel_plans)

            if cache.episode_outdir is not None:
                json.dump({
                    "instruction": instruction, "context": None,
                    "plans": highlevel_plans, 'parsed_plans': cache.highlevel_plans,
                }, open(os.path.join(cache.episode_outdir, 'highlevel_plans.json'), 'w'))

        # print('highlevel step id', self.cache.highlevel_step_id)
        if cache.highlevel_step_id >= len(cache.highlevel_plans):
            if self.config.pipeline.restart:
                # print('-------restart from the begining-----')
                cache.highlevel_step_id = 0
                cache.valid_actions = []
                cache.object_vars = {}
                cache.ret_objs = {}
                cache.grasped_obj_name = None, 
                cache.prev_ee_pose = copy.deepcopy(obs_state_dict['gripper'])
            else:
                return {'action': np.zeros((8, )), 'cache': cache}

        plan = cache.highlevel_plans[cache.highlevel_step_id]
        if plan is None:
            return {'action': np.zeros((8, )), 'cache': cache}
        
        if plan['action'] == 'release':
            action = gripper_pose
            action[7] = 1
            cache.highlevel_step_id += 1
            cache.grasped_obj_name = None
            return {'action': action, 'cache': cache}

        vlm_results = self.vlm_pipeline.run(
            rgb_images, pcd_images, arm_links_info
        )

        if plan.is_target_variable and plan.target in cache.ret_objs:
            target_var_xyz = cache.ret_objs[plan.target]
        else:
            target_var_xyz = None

        zrange = None
        if plan['object'] is not None and 'drawer' in plan['object']:
            obj_height = np.concatenate(
                [obj.pcd_xyz[:, 2] for obj in vlm_results.objects \
                    if len(obj.captions)==0 or obj.captions[0] != 'robot'], 0
            )
            # TODO: there are some noisy robot points
            obj_height = np.percentile(obj_height, 99) - obj_height.min()
            # print('obj_height', obj_height, 'obj_name', plan['object'])
            zrange = self.llm_planner.estimate_height_range(plan['object'], obj_height)
            # print('zrange', zrange)
            if zrange is not None:
                zrange += self.workspace['TABLE_HEIGHT']
        # print(plan)
        if plan['target'] is not None and 'safe' in task_str and ('safe' in plan['target'] or 'shelf' in plan['target']):
            obj_height = np.concatenate(
                [obj.pcd_xyz[:, 2] for obj in vlm_results.objects \
                    if len(obj.captions)==0 or obj.captions[0] != 'robot'], 0
            )
            # TODO: there are some noisy robot points
            obj_height = np.percentile(obj_height, 99) - obj_height.min()
            # print('obj_height', obj_height, 'obj_name', plan['target'])
            zrange = self.llm_planner.estimate_height_range(plan['target'], obj_height)
            # print('zrange', zrange)
            if zrange is not None:
                zrange += self.workspace['TABLE_HEIGHT']
            
        batch, extra_outs = self.prepare_motion_planner_input(
            vlm_results, plan,
            arm_links_info, gripper_pose, 
            voxel_size=self.mp_config.MODEL.action_config.voxel_size,
            rm_robot=self.mp_config.TRAIN_DATASET.rm_robot,
            num_points=self.mp_config.TRAIN_DATASET.num_points,
            same_npoints_per_example=self.mp_config.TRAIN_DATASET.same_npoints_per_example,
            xyz_shift=self.mp_config.TRAIN_DATASET.xyz_shift,
            xyz_norm=self.mp_config.TRAIN_DATASET.xyz_norm,
            use_height=self.mp_config.TRAIN_DATASET.use_height,
            use_color=self.mp_config.TRAIN_DATASET.get('use_color', False),
            instr_include_objects=self.mp_config.TRAIN_DATASET.get('instr_include_objects', False),
            zrange=zrange,
            target_var_xyz=target_var_xyz,
        )

        if 'mani_obj' in extra_outs:
            cache.ret_objs[extra_outs.mani_obj.name] = extra_outs.mani_obj.pcd_xyz
            if plan['action'] == 'grasp':
                cache.grasped_obj_name = extra_outs.mani_obj.name
        
        pred_actions = self.motion_planner(batch, compute_loss=False)[0] # (max_action_len, 8)
        pred_actions[:, 7:] = torch.sigmoid(pred_actions[:, 7:])
        pred_actions = pred_actions.data.cpu().numpy()
        # print(pred_actions)

        # rescale the predicted position
        pred_actions[:, :3] = (pred_actions[:, :3] * batch['pc_radius']) + batch['pc_centroids']
        pred_actions[:, 2] = np.maximum(pred_actions[:, 2], self.workspace['TABLE_HEIGHT'] + 0.005)

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
        
        cache.valid_actions = valid_actions[1:]
        # print('valid actions', len(valid_actions), valid_actions)
        out = {'action': valid_actions[0][:8]}

        if cache.episode_outdir is not None:
            del batch['txt_embeds'], batch['txt_lens']
            np.save(
                os.path.join(cache.episode_outdir, f'{step_id}.npy'),
                {
                    'batch': {k: v.data.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in batch.items()},
                    'obs': obs_state_dict,
                    'valid_actions': valid_actions,
                }
            )

        if cache.grasped_obj_name is not None and \
            cache.grasped_obj_name in cache.ret_objs and \
            plan['action'].startswith('move grasped object'):
            # rotate the grasped object position
            self.move_grasped_obj_xyz(out['action'], cache.prev_ee_pose, cache.ret_objs[cache.grasped_obj_name])
        cache.prev_ee_pose = out['action']

        out['cache'] = cache
        return out
    
