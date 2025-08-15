import os
import numpy as np
import json
from tqdm import tqdm
import argparse

import lmdb
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

import open3d as o3d

from genrobo3d.configs.rlbench.constants import get_robot_workspace
from genrobo3d.utils.point_cloud import voxelize_pcd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir')
    parser.add_argument('--output_dir')
    parser.add_argument('--taskvar_file', default=None)
    parser.add_argument('--task', default=None, type=str)
    parser.add_argument('--variation_id', default=None, type=int)
    parser.add_argument('--voxel_size', type=float, default=0.01, help='meters')
    parser.add_argument('--real_robot', default=False, action='store_true')
    parser.add_argument('--cam_ids', default=None, type=int, nargs='+', help='use all by default')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    if args.task is not None and args.variation_id is not None:
        taskvars = [f'{args.task}+{args.variation_id}']
    elif args.taskvar_file is not None:
        taskvars = json.load(open(args.taskvar_file))
    else:
        taskvars = [x for x in os.listdir(args.input_dir) if '+' in x]

    workspace = get_robot_workspace(real_robot=args.real_robot)
    if args.cam_ids is None:
        cam_ids = [0, 1, 2, 3] # use all cameras
    else:
        cam_ids = args.cam_ids

    for taskvar in tqdm(taskvars):
        input_lmdb_dir = os.path.join(args.input_dir, taskvar)
        if not os.path.exists(input_lmdb_dir):
            print(taskvar, 'not exists')
            continue
        if os.path.exists(os.path.join(args.output_dir, taskvar)):
            continue
            
        out_lmdb_env = lmdb.open(os.path.join(args.output_dir, taskvar), map_size=int(1024**4))

        with lmdb.open(input_lmdb_dir, readonly=True, lock=False) as lmdb_env:
            with lmdb_env.begin() as txn:
                for key, value in txn.cursor():
                    value = msgpack.unpackb(value)
                    
                    rgb = value['rgb'][:, cam_ids] # (T, N_cam, H, W, 3)
                    pc = value['pc'][:, cam_ids]   # (T, N_cam, H, W, 3)
                    if 'mask' in value:
                        sem = value['mask'][:, cam_ids] # (T, N_cam, H, W)
                    elif 'gt_masks' in value:
                        sem = value['gt_masks'][:, cam_ids] # (T, N_cam, H, W)
                    else:
                        sem = None

                    outs = {
                        'xyz': [], 'rgb': [], 'sem': []
                    }
                    for value_key in ['bbox_info', 'pose_info', 'key_frameids', 'action']:
                        if value_key in value:
                            outs[value_key] = value[value_key]
                    
                    num_steps = rgb.shape[0]
                    for t in range(num_steps):
                        t_pc = pc[t].reshape(-1, 3)
                        in_mask = (t_pc[:, 0] > workspace['X_BBOX'][0]) & (t_pc[:, 0] < workspace['X_BBOX'][1]) & \
                                (t_pc[:, 1] > workspace['Y_BBOX'][0]) & (t_pc[:, 1] < workspace['Y_BBOX'][1]) & \
                                (t_pc[:, 2] > workspace['Z_BBOX'][0]) & (t_pc[:, 2] < workspace['Z_BBOX'][1])
                        if args.real_robot:
                            in_mask = in_mask & (t_pc[:, 2] > workspace['TABLE_HEIGHT'])
                        t_pc = t_pc[in_mask]
                        t_rgb = rgb[t].reshape(-1, 3)[in_mask]
                        if sem is not None:
                            t_sem = sem[t].reshape(-1)[in_mask]
                        
                        t_pc, mask = voxelize_pcd(t_pc, voxel_size=args.voxel_size)
                        t_rgb = t_rgb[mask]
                        if sem is not None:
                            t_sem = t_sem[mask]

                        if args.real_robot:
                            # remove point cloud outliers for noisy real point clouds
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(t_pc)
                            pcd.colors = o3d.utility.Vector3dVector(t_rgb)
                            pcd, outlier_masks = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.2)
                            t_pc = t_pc[outlier_masks]
                            t_rgb = t_rgb[outlier_masks]
                            if sem is not None:
                                t_sem = t_sem[outlier_masks]

                        outs['xyz'].append(t_pc)
                        outs['rgb'].append(t_rgb)
                        if sem is not None:
                            outs['sem'].append(t_sem)

                    if len(outs['sem']) == 0:
                        del outs['sem']

                    out_txn = out_lmdb_env.begin(write=True)
                    out_txn.put(key, msgpack.packb(outs))
                    out_txn.commit()
                        
        out_lmdb_env.close()

if __name__ == '__main__':
    main()
