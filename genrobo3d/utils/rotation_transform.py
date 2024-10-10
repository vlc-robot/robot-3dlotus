import torch

from scipy.spatial.transform import Rotation as R
import numpy as np


class RotationMatrixTransform():
    # https://github.com/papagina/RotationContinuity/blob/758b0ce551c06372cab7022d4c0bdf331c89c696/shapenet/code/tools.py
    
    @staticmethod
    def normalize_vector(v):
        '''
        Args:
            v: torch.Tensor, (batch, n)
        Returns:
            normalized v: torch.Tensor, (batch, n)
        '''
        device = v.device
        batch = v.shape[0]
        v_mag = torch.sqrt(v.pow(2).sum(1))# batch
        v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).to(device)))
        v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
        v = v / v_mag
        return v

    @staticmethod
    def cross_product(u, v):
        '''
        Args:
            u: torch.Tensor, (batch, 3)
            v: torch.Tensor, (batch, 3)
        Returns:
            u x v: torch.Tensor, (batch, 3)
        '''
        batch = u.shape[0]
        i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
        j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
        k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
            
        out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)    
        return out

    @staticmethod
    def compute_rotation_matrix_from_ortho6d(poses):
        '''
        Args:
            poses: torch.Tensor, (batch, 6)
        Returns:
            matrix: torch.Tensor, (batch, 3, 3)
        '''
        x_raw = poses[:, 0:3]    # (batch, 3)
        y_raw = poses[:, 3:6]    # (batch, 3)
            
        x = RotationMatrixTransform.normalize_vector(x_raw) # (batch, 3)
        z = RotationMatrixTransform.cross_product(x, y_raw) # (batch, 3)
        z = RotationMatrixTransform.normalize_vector(z)     # (batch, 3)
        y = RotationMatrixTransform.cross_product(z, x)     # (batch, 3)
            
        x = x.view(-1, 3, 1)
        y = y.view(-1, 3, 1)
        z = z.view(-1, 3, 1)
        
        matrix = torch.cat((x, y, z), 2) # (batch, 3)
        return matrix

    @staticmethod
    def get_ortho6d_from_rotation_matrix(matrix):
        '''
        Args:
            matrix: torch.Tensor, (batch, 3, 3)
        Returns:
            vector: torch.Tensor, (batch, 6)
        '''
        # The orhto6d represents the first two column vectors a1 and a2 of the
        # rotation matrix: [ | , |,  | ]
        #                  [ a1, a2, a3]
        #                  [ | , |,  | ]
        ortho6d = matrix[:, :, :2].permute(0, 2, 1).flatten(-2)
        return ortho6d

    @staticmethod
    def quaternion_to_matrix(quats):
        '''
        Args:
            quats: np.array/torch.Tensor, (batch, 4), xyzw (scalar-last convention)
        Returns:
            matrix: (batch, 3, 3)
        '''
        mats = [R.from_quat(q).as_matrix() for q in quats]
        mats = np.stack(mats, 0)
        if isinstance(quats, torch.Tensor):
            mats = torch.from_numpy(mats).to(quats.device)
        return mats

    @staticmethod
    def matrix_to_quaternion(mats):
        '''
        Args:
            mat: np.array/torch.Tensor, (batch, 3, 3)
        Returns:
            quats: (batch, 4), xyzw (scalar-last convention)
        '''
        quats = [R.from_matrix(m).as_quat() for m in mats]
        quats = np.stack(quats, 0)
        if isinstance(mats, torch.Tensor):
            quats = torch.from_numpy(quats).to(mats.device)
        return quats
    
    @staticmethod
    def quaternion_to_ortho6d(quats):
        return RotationMatrixTransform.get_ortho6d_from_rotation_matrix(
            RotationMatrixTransform.quaternion_to_matrix(quats)
        )
    
    @staticmethod
    def ortho6d_to_quaternion(ortho6d):
        return RotationMatrixTransform.matrix_to_quaternion(
            RotationMatrixTransform.compute_rotation_matrix_from_ortho6d(ortho6d)
        )
    
    @staticmethod
    def quaternion_to_euler(quats):
        '''
        Args:
            quats: (batch, 4), xyzw (scalar-last convention)
        Returns:
            eulers: (batch, 3), [-180, 180]
        '''
        eulers = [R.from_quat(q).as_euler('xyz', degrees=True) for q in quats]
        eulers = np.stack(eulers, 0)
        if isinstance(quats, torch.Tensor):
            eulers = torch.from_numpy(eulers).to(quats.device)
        return eulers
    
    @staticmethod
    def euler_to_quaternion(eulers):
        '''
        Args:
            eulers: (batch, 3), [-180, 180]
        Returns:
            quats: (batch, 4), xyzw (scalar-last convention)
        '''
        quats = [R.from_euler('xyz', e, degrees=True).as_quat() for e in eulers]
        quats = np.stack(quats, 0)
        quats = torch.from_numpy(quats).to(eulers.device)
        return quats
    

################# functions from RVT-2 #################

def sensitive_gimble_fix(euler):
    """
    :param euler: euler angles in degree as np.ndarray in shape either [3] or
    [b, 3]
    """
    # selecting sensitive angle
    select1 = (89 < euler[..., 1]) & (euler[..., 1] < 91)
    euler[select1, 1] = 90
    # selecting sensitive angle
    select2 = (-91 < euler[..., 1]) & (euler[..., 1] < -89)
    euler[select2, 1] = -90

    # recalulating the euler angles, see assert
    r = R.from_euler("xyz", euler, degrees=True)
    euler = r.as_euler("xyz", degrees=True)

    select = select1 | select2
    assert (euler[select][..., 2] == 0).all(), euler

    return euler

def quaternion_to_discrete_euler(quaternion, resolution, gimble_fix=True):
    """
    :param gimble_fix: the euler values for x and y can be very sensitive
        around y=90 degrees. this leads to a multimodal distribution of x and y
        which could be hard for a network to learn. When gimble_fix is true, around
        y=90, we change the mode towards x=0, potentially making it easy for the
        network to learn.
    """
    r = R.from_quat(quaternion)

    euler = r.as_euler("xyz", degrees=True)
    if gimble_fix:
        euler = sensitive_gimble_fix(euler)

    euler += 180
    assert np.min(euler) >= 0 and np.max(euler) <= 360
    disc = np.around((euler / resolution)).astype(int)
    disc[disc == int(360 / resolution)] = 0
    return disc

def discrete_euler_to_quaternion(discrete_euler, resolution):
    euler = (discrete_euler * resolution) - 180
    return R.from_euler("xyz", euler, degrees=True).as_quat()