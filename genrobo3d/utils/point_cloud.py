from typing import Tuple

import open3d as o3d

import numpy as np


def voxelize_pcd(xyz, voxel_size=0.005):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    # downsampling
    min_bound = np.min(xyz, 0)
    max_bound = np.max(xyz, 0)
    pcd, _, trace = pcd.voxel_down_sample_and_trace(voxel_size, min_bound, max_bound)
    xyz = np.asarray(pcd.points).astype(np.float32)
    trace = np.array([v[0] for v in trace])
    return xyz, trace

def get_pc_foreground_mask(xyz, workspace):
    mask = (xyz[:, 0] > workspace['X_BBOX'][0]) & (xyz[:, 0] < workspace['X_BBOX'][1]) \
           & (xyz[:, 1] > workspace['Y_BBOX'][0]) & (xyz[:, 1] < workspace['Y_BBOX'][1]) \
           & (xyz[:, 2] > workspace['Z_BBOX'][0]) & (xyz[:, 2] < workspace['Z_BBOX'][1])
    mask = mask & (xyz[:, 2] > workspace['TABLE_HEIGHT'])
    return mask

def convert_gripper_pose_world_to_image(obs, camera: str) -> Tuple[int, int]:
    '''Convert the gripper pose from world coordinate system to image coordinate system.
    image[v, u] is the gripper location.
    '''
    extrinsics_44 = obs.misc[f"{camera}_camera_extrinsics"].astype(np.float32)
    extrinsics_44 = np.linalg.inv(extrinsics_44)

    intrinsics_33 = obs.misc[f"{camera}_camera_intrinsics"].astype(np.float32)
    intrinsics_34 = np.concatenate([intrinsics_33, np.zeros((3, 1), dtype=np.float32)], 1)

    gripper_pos_31 = obs.gripper_pose[:3].astype(np.float32)[:, None]
    gripper_pos_41 = np.concatenate([gripper_pos_31, np.ones((1, 1), dtype=np.float32)], 0)

    points_cam_41 = extrinsics_44 @ gripper_pos_41

    proj_31 = intrinsics_34 @ points_cam_41
    proj_3 = proj_31[:, 0]

    u = int((proj_3[0] / proj_3[2]).round())
    v = int((proj_3[1] / proj_3[2]).round())

    return u, v
