from typing import List, Tuple

import numpy as np

import torch
import einops
import json
from scipy.spatial.transform import Rotation as R


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

class PointWorld2Image(object):
    def __init__(self, camera_param_file):
        self.camera_params = json.load(open(camera_param_file, 'r'))
        for k, v in self.camera_params.items():
            if isinstance(v, list):
                self.camera_params[k] = np.array(v, dtype=np.float32)
                
        self.cameras = []
        for k, v in self.camera_params.items():
            if k.endswith('_extrinsics'):
                self.cameras.append('_'.join(k.split('_')[:-2]))
                
        self.camera_transform = {}
        for camera in self.cameras:
            extrinsics_44 = self.camera_params[f"{camera}_camera_extrinsics"]
            extrinsics_44 = np.linalg.inv(extrinsics_44)

            intrinsics_33 = self.camera_params[f"{camera}_camera_intrinsics"]
            intrinsics_34 = np.concatenate([intrinsics_33, np.zeros((3, 1), dtype=np.float32)], 1)
            
            self.camera_transform[camera] = torch.from_numpy(intrinsics_34 @ extrinsics_44).float()

    def __call__(self, cameras, points, return_float=False):
        '''Convert point from world coordinate system to image coordinate system.
        image[v, u] is the point location.
        points: torch.FloatTensor (batch, 3, npoints)
        '''
        batch_size, _, npoints = points.size()
        device = points.device
        points_31 = einops.rearrange(points, 'b c n -> c (b n)')
        points_41 = torch.cat([points_31, torch.ones(1, points_31.size(-1)).float().to(device)], 0)

        outs = []
        for camera in cameras:
            projs_31 = torch.matmul(self.camera_transform[camera], points_41)

            u = projs_31[0] / projs_31[2]
            if not return_float:
                u = u.round().long()
            u = einops.rearrange(u, '(b n) -> b n', b=batch_size, n=npoints)
            
            v = (projs_31[1] / projs_31[2])
            if not return_float:
                v = v.round().long()
            v = einops.rearrange(v, '(b n) -> b n', b=batch_size, n=npoints)
            
            # (b, 2, npoints)
            outs.append(torch.stack([v, u], dim=1))
        
        return outs
        

def quaternion_to_discrete_euler(quaternion, resolution: int):
    euler = R.from_quat(quaternion).as_euler('xyz', degrees=True) + 180
    assert np.min(euler) >= 0 and np.max(euler) <= 360
    disc = np.around((euler / resolution)).astype(int)
    disc[disc == int(360 / resolution)] = 0
    return disc

def discrete_euler_to_quaternion(discrete_euler, resolution: int):
    euluer = (discrete_euler * resolution) - 180
    return R.from_euler('xyz', euluer, degrees=True).as_quat()

def euler_to_quat(euler, degrees):
    rotation = R.from_euler("xyz", euler, degrees=degrees)
    return rotation.as_quat()

def quat_to_euler(quat, degrees):
    rotation = R.from_quat(quat)
    return rotation.as_euler("xyz", degrees=degrees)