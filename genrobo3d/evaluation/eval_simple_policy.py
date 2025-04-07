from typing import Tuple, Dict, List

import os
import json
import jsonlines
import tap
import copy
from pathlib import Path
from filelock import FileLock

import torch
import numpy as np
from scipy.special import softmax

# TODO: error when import in a different order: Error /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.34’ not found or /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found
# TODO: always import torch first
import open3d as o3d
from sklearn.neighbors import LocalOutlierFactor
from scipy.spatial.transform import Rotation as R

from genrobo3d.train.utils.misc import set_random_seed
from genrobo3d.configs.default import get_config

try:
    from genrobo3d.rlbench.environments import RLBenchEnv
except:
    print('No RLBench')

from genrobo3d.train.train_simple_policy import MODEL_FACTORY

from genrobo3d.configs.rlbench.constants import get_robot_workspace, get_rlbench_labels
from genrobo3d.utils.robot_box import RobotBox
from genrobo3d.train.datasets.common import gen_seq_masks
from genrobo3d.evaluation.common import write_to_file
from genrobo3d.vlm_models.clip_encoder import ClipEncoder


class Arguments(tap.Tap):
    exp_config: str
    device: str = 'cuda'  # cpu, cuda

    microstep_data_dir: str = ''
    seed: int = 100  # seed for RLBench
    num_demos: int = 20
    taskvar: str = 'push_button+0'
    checkpoint: str = None

    headless: bool = False
    max_tries: int = 10
    max_steps: int = 25
    cam_rand_factor: float = 0.0
    image_size: List[int] = [256, 256]

    save_image: bool = False
    save_obs_outs_dir: str = None
    record_video: bool = False
    not_include_robot_cameras: bool = False
    video_rotate_cam: bool = False
    video_resolution: int = 480

    num_ensembles: int = 1

    best_disc_pos: str = 'max' # max, ens1

    real_robot: bool = False

class Actioner(object):
    def __init__(self, args) -> None:
        self.args = args
        if self.args.save_obs_outs_dir is not None:
            os.makedirs(self.args.save_obs_outs_dir, exist_ok=True)

        self.WORKSPACE = get_robot_workspace(real_robot=args.real_robot)
        self.device = torch.device(args.device)

        config = get_config(args.exp_config, args.remained_args)
        self.config = config
        self.config.defrost()
        self.config.TRAIN_DATASET.sample_points_by_distance = self.config.TRAIN_DATASET.get('sample_points_by_distance', False)
        self.config.TRAIN_DATASET.rm_pc_outliers = self.config.TRAIN_DATASET.get('rm_pc_outliers', False)
        self.config.TRAIN_DATASET.rm_pc_outliers_neighbors = self.config.TRAIN_DATASET.get('rm_pc_outliers_neighbors', 10)
        self.config.TRAIN_DATASET.same_npoints_per_example = self.config.TRAIN_DATASET.get('same_npoints_per_example', False)
        self.config.MODEL.action_config.best_disc_pos = args.best_disc_pos

        if args.checkpoint is not None:
            config.checkpoint = args.checkpoint

        model_class = MODEL_FACTORY[config.MODEL.model_class]
        self.model = model_class(config.MODEL)
        if config.checkpoint:
            checkpoint = torch.load(
                config.checkpoint, map_location=lambda storage, loc: storage
            )
            self.model.load_state_dict(checkpoint, strict=True)

        self.model.to(self.device)
        self.model.eval()

        self.config.freeze()

        data_cfg = self.config.TRAIN_DATASET
        self.data_cfg = data_cfg
        if os.path.exists(data_cfg.instr_embed_file):
            self.instr_embeds = np.load(data_cfg.instr_embed_file, allow_pickle=True).item()
        else:
            self.instr_embeds = {}
        if data_cfg.instr_embed_type == 'last':
            self.instr_embeds = {instr: embeds[-1:] for instr, embeds in self.instr_embeds.items()}
        self.taskvar_instrs = json.load(open(data_cfg.taskvar_instr_file))

        # Load clip text encoder
        self.clip_model = ClipEncoder()

        self.TABLE_HEIGHT = self.WORKSPACE['TABLE_HEIGHT']

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
            env_name='real' if self.args.real_robot else 'rlbench'
        )
        _, robot_point_ids = robot_box.get_pc_overlap_ratio(xyz=xyz, return_indices=True)
        robot_point_ids = np.array(list(robot_point_ids))
        mask = np.ones((xyz.shape[0], ), dtype=bool)
        if len(robot_point_ids) > 0:
            mask[robot_point_ids] = False
        return mask
    
    def _rm_pc_outliers(self, xyz, rgb=None):
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(xyz)
        # pcd, idxs = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
        # pcd, idxs = pcd.remove_radius_outlier(nb_points=16, radius=0.03)
        clf = LocalOutlierFactor(n_neighbors=self.data_cfg.rm_pc_outliers_neighbors)
        preds = clf.fit_predict(xyz)
        idxs = (preds == 1)
        xyz = xyz[idxs]
        if rgb is not None:
            rgb = rgb[idxs]
        return xyz, rgb
    
    def process_point_clouds(
        self, xyz, rgb, gt_sem=None, ee_pose=None, arm_links_info=None, taskvar=None
    ):
        # keep points in robot workspace
        xyz = xyz.reshape(-1, 3)
        in_mask = (xyz[:, 0] > self.WORKSPACE['X_BBOX'][0]) & (xyz[:, 0] < self.WORKSPACE['X_BBOX'][1]) & \
                  (xyz[:, 1] > self.WORKSPACE['Y_BBOX'][0]) & (xyz[:, 1] < self.WORKSPACE['Y_BBOX'][1]) & \
                  (xyz[:, 2] > self.WORKSPACE['Z_BBOX'][0]) & (xyz[:, 2] < self.WORKSPACE['Z_BBOX'][1])
        if self.data_cfg.rm_table:
            in_mask = in_mask & (xyz[:, 2] > self.WORKSPACE['TABLE_HEIGHT'])
        xyz = xyz[in_mask]
        rgb = rgb.reshape(-1, 3)[in_mask]
        if gt_sem is not None:
            gt_sem = gt_sem.reshape(-1)[in_mask]

        # downsampling
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd, _, trace = pcd.voxel_down_sample_and_trace(
            self.config.MODEL.action_config.voxel_size, np.min(xyz, 0), np.max(xyz, 0)
        )
        xyz = np.asarray(pcd.points)
        trace = np.array([v[0] for v in trace])
        rgb = rgb[trace]
        if gt_sem is not None:
            gt_sem = gt_sem[trace]

        if self.args.real_robot:
            for _ in range(1):
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz)
                pcd.colors = o3d.utility.Vector3dVector(rgb)
                pcd, outlier_masks = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.2)
                xyz = xyz[outlier_masks]
                rgb = rgb[outlier_masks]
                if gt_sem is not None:
                    gt_sem = gt_sem[outlier_masks]

        # remove non-object points
        if not self.args.real_robot:
            rm_label_ids = get_rlbench_labels(
                taskvar.split('+')[0], table=self.data_cfg.rm_table, robot=(self.data_cfg.rm_robot=='gt'), wall=False, floor=False
            )
            if len(rm_label_ids) > 0:
                rm_mask = self._get_mask_with_label_ids(gt_sem, rm_label_ids)
                xyz = xyz[~rm_mask]
                rgb = rgb[~rm_mask]
        
        if self.data_cfg.rm_robot.startswith('box'):
            mask = self._get_mask_with_robot_box(xyz, arm_links_info, self.data_cfg.rm_robot)
            xyz = xyz[mask]
            rgb = rgb[mask]

        if self.data_cfg.rm_pc_outliers:
            xyz, rgb = self._rm_pc_outliers(xyz, rgb)

        # sampling points
        if len(xyz) > self.data_cfg.num_points:
            if self.data_cfg.sample_points_by_distance:
                dists = np.sqrt(np.sum((xyz - ee_pose[:3])**2, 1))
                probs = 1 / np.maximum(dists, 0.1)
                probs = np.maximum(softmax(probs), 1e-30) 
                probs = probs / sum(probs)
                # probs = 1 / dists
                # probs = probs / np.sum(probs)
                point_idxs = np.random.choice(len(xyz), self.data_cfg.num_points, replace=False, p=probs)
            else:
                point_idxs = np.random.choice(len(xyz), self.data_cfg.num_points, replace=False)
        else:
            if self.data_cfg.same_npoints_per_example:
                point_idxs = np.random.choice(xyz.shape[0], self.data_cfg.num_points, replace=True)
            else:
                point_idxs = np.arange(xyz.shape[0])
        xyz = xyz[point_idxs]
        rgb = rgb[point_idxs]
        height = xyz[:, -1] - self.TABLE_HEIGHT

        # normalize
        if self.data_cfg.xyz_shift == 'none':
            centroid = np.zeros((3, ))
        elif self.data_cfg.xyz_shift == 'center':
            centroid = np.mean(xyz, 0)
        elif self.data_cfg.xyz_shift == 'gripper':
            centroid = copy.deepcopy(ee_pose[:3])
        if self.data_cfg.xyz_norm:
            radius = np.max(np.sqrt(np.sum((xyz - centroid) ** 2, axis=1)))
        else:
            radius = 1

        xyz = (xyz - centroid) / radius
        height = height / radius
        ee_pose[:3] = (ee_pose[:3] - centroid) / radius
        
        rgb = (rgb / 255.) * 2 - 1
        pc_ft = np.concatenate([xyz, rgb], 1)
        if self.data_cfg.get('use_height', False):
            pc_ft = np.concatenate([pc_ft, height[:, None]], 1)

        return pc_ft, centroid, radius, ee_pose


    def preprocess_obs(self, taskvar, step_id, obs):
        rgb = np.stack(obs['rgb'], 0)  # (N, H, W, C)
        xyz = np.stack(obs['pc'], 0)  # (N, H, W, C)
        if 'gt_mask' in obs:
            gt_sem = np.stack(obs['gt_mask'], 0)  # (N, H, W) 
        else:
            gt_sem = None
        
        # select one instruction
        instr = self.taskvar_instrs[taskvar][0]
        if instr not in self.instr_embeds:
            instr_embed = self.clip_model('text', instr, use_prompt=False, output_hidden_states=True)
            instr_embed = instr_embed[0].data.cpu().numpy()
            if self.data_cfg.instr_embed_type == 'last':
                instr_embed = instr_embed[-1:]
            self.instr_embeds[instr] = instr_embed
        instr_embed = self.instr_embeds[instr]
        
        pc_ft, pc_centroid, pc_radius, ee_pose = self.process_point_clouds(
            xyz, rgb, gt_sem=gt_sem, ee_pose=copy.deepcopy(obs['gripper']), 
            arm_links_info=obs['arm_links_info'], taskvar=taskvar
        )
        
        batch = {
            'pc_fts': torch.from_numpy(pc_ft).float(),
            'pc_centroids': pc_centroid,
            'pc_radius': pc_radius,
            'ee_poses': torch.from_numpy(ee_pose).float().unsqueeze(0),
            'step_ids': torch.LongTensor([step_id]),
            'txt_embeds': torch.from_numpy(instr_embed).float(),
            'txt_lens': [instr_embed.shape[0]],
            'npoints_in_batch': [pc_ft.shape[0]],
            'offset': torch.LongTensor([pc_ft.shape[0]]),
        }
        if self.config.MODEL.model_class == 'SimplePolicyPCT':
            batch['pc_fts'] = batch['pc_fts'].unsqueeze(0)
            batch['txt_masks'] = torch.from_numpy(
                gen_seq_masks(batch['txt_lens'])
            ).bool()
            batch['txt_embeds'] = batch['txt_embeds'].unsqueeze(0)
            
        # for k, v in batch.items():
        #     if k not in ['pc_centroids', 'pc_radius', 'npoints_in_batch']:
        #         print(k, v.size())
        return batch

    def predict(
        self, task_str=None, variation=None, step_id=None, obs_state_dict=None, 
        episode_id=None, instructions=None,
    ):
        # print(obs_state_dict)
        taskvar = f'{task_str}+{variation}'
        batch = self.preprocess_obs(
            taskvar, step_id, obs_state_dict,
        )
        with torch.no_grad():
            actions = []
            # TODO
            for _ in range(self.args.num_ensembles):
                action = self.model(batch)[0].data.cpu()
                actions.append(action)
            if len(actions) > 1:
                # print(torch.stack(actions, 0))
                avg_action = torch.stack(actions, 0).mean(0)
                pred_rot = torch.from_numpy(R.from_euler(
                    'xyz', np.mean([R.from_quat(x[3:-1]).as_euler('xyz') for x in actions], 0),
                ).as_quat())
                action = torch.cat([avg_action[:3], pred_rot, avg_action[-1:]], 0)
            else:
                action = actions[0]
        action[-1] = torch.sigmoid(action[-1]) > 0.5
        
        # action = action.data.cpu().numpy()
        action = action.numpy()
        action[:3] = action[:3] * batch['pc_radius'] + batch['pc_centroids']
        # TODO: ensure the action height is above the table
        action[2] = max(action[2], self.TABLE_HEIGHT+0.005)

        out = {
            'action': action
        }

        if self.args.save_obs_outs_dir is not None:
            np.save(
                os.path.join(self.args.save_obs_outs_dir, f'{task_str}+{variation}-{episode_id}-{step_id}.npy'),
                {
                    'batch': {k: v.data.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in batch.items()},
                    'obs': obs_state_dict,
                    'action': action
                }
            )

        return out


def evaluate_actioner(args):    
    
    set_random_seed(args.seed)

    actioner = Actioner(args)
    
    pred_dir = os.path.join(actioner.config.output_dir, 'preds', f'seed{args.seed}')
    if args.cam_rand_factor > 0:
        pred_dir = '%s-cam_rand_factor%.1f' % (pred_dir, args.cam_rand_factor)
    os.makedirs(pred_dir, exist_ok=True)

    if len(args.image_size) == 1:
        args.image_size = [args.image_size[0], args.image_size[0]]    # (height, width)

    outfile = os.path.join(pred_dir, 'results.jsonl')

    existed_data = set()
    if os.path.exists(outfile):
        with jsonlines.open(outfile, 'r') as f:
            for item in f:
                existed_data.add((item['checkpoint'], '%s+%d'%(item['task'], item['variation'])))

    if (args.checkpoint, args.taskvar) in existed_data:
        return

    env = RLBenchEnv(
        data_path=args.microstep_data_dir,
        apply_rgb=True,
        apply_pc=True,
        apply_mask=True,
        headless=args.headless,
        image_size=args.image_size,
        cam_rand_factor=args.cam_rand_factor,
    )

    task_str, variation = args.taskvar.split('+')
    variation = int(variation)

    if args.microstep_data_dir != '':
        episodes_dir = os.path.join(args.microstep_data_dir, task_str, f"variation{variation}", "episodes")
        demo_keys, demos = [], []
        if os.path.exists(str(episodes_dir)):
            episode_ids = os.listdir(episodes_dir)
            episode_ids.sort(key=lambda ep: int(ep[7:]))
            for idx, ep in enumerate(episode_ids):
                # episode_id = int(ep[7:])
                try:
                    demo = env.get_demo(task_str, variation, idx, load_images=False)
                    demo_keys.append(f'episode{idx}')
                    demos.append(demo)
                except Exception as e:
                    print('\tProblem to load demo_id:', idx, ep)
                    print(e)
    else:
        demo_keys = None
        demos = None
            
    success_rate = env.evaluate(
        task_str, variation,
        actioner=actioner,
        max_episodes=args.max_steps,
        num_demos=len(demos) if demos is not None else args.num_demos,
        demos=demos,
        demo_keys=demo_keys,
        log_dir=Path(pred_dir),
        max_tries=args.max_tries,
        save_image=args.save_image,
        record_video=args.record_video,
        include_robot_cameras=(not args.not_include_robot_cameras),
        video_rotate_cam=args.video_rotate_cam,
        video_resolution=args.video_resolution,
    )

    print("Testing Success Rate {}: {:.04f}".format(task_str, success_rate))
    write_to_file(
        outfile,
        {
            'checkpoint': args.checkpoint,
            'task': task_str, 'variation': variation,
            'num_demos': args.num_demos, 'sr': success_rate
        }
    )



if __name__ == '__main__':
    args = Arguments().parse_args(known_only=True)
    args.remained_args = args.extra_args
    
    evaluate_actioner(args)
