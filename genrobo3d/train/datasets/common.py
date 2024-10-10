import numpy as np
import random
import torch
import copy

def pad_tensors(tensors, lens=None, pad=0, max_len=None):
    """B x [T, ...] torch tensors"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
    max_len = max(lens) if max_len is None else max_len
    bs = len(tensors)
    hid = list(tensors[0].size()[1:])
    size = [bs, max_len] + hid

    dtype = tensors[0].dtype
    output = torch.zeros(*size, dtype=dtype)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
    return output

def gen_seq_masks(seq_lens, max_len=None):
    """
    Args:
        seq_lens: list or nparray int, shape=(N, )
    Returns:
        masks: nparray, shape=(N, L), padded=0
    """
    seq_lens = np.array(seq_lens)
    if max_len is None:
        max_len = max(seq_lens)
    if max_len == 0:
        return np.zeros((len(seq_lens), 0), dtype=bool)
    batch_size = len(seq_lens)
    masks = np.arange(max_len).reshape(-1, max_len).repeat(batch_size, 0)
    masks = masks < seq_lens.reshape(-1, 1)
    return masks


def normalize_pc(pc, centroid=None, return_params=False):
    # Normalize the point cloud to [-1, 1]
    if centroid is None:
        centroid = np.mean(pc, axis=0)
    else:
        centroid = copy.deepcopy(centroid)
    
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    if m < 1e-6:
        pc = np.zeros_like(pc)
    else:
        pc = pc / m
    if return_params:
        return pc, (centroid, m)
    return pc

def random_scale_pc(pc, scale_low=0.8, scale_high=1.25):
    # Randomly scale the point cloud.
    scale = np.random.uniform(scale_low, scale_high)
    pc = pc * scale
    return pc

def shift_pc(pc, shift_range=0.1):
    # Randomly shift point cloud.
    shift = np.random.uniform(-shift_range, shift_range, size=[3])
    pc = pc + shift
    return pc

def rotate_perturbation_pc(pc, angle_sigma=0.06, angle_clip=0.18):
    # Randomly perturb the point cloud by small rotations (unit: radius)
    angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
    cosval, sinval = np.cos(angles), np.sin(angles)
    Rx = np.array([[1, 0, 0], [0, cosval[0], -sinval[0]], [0, sinval[0], cosval[0]]])
    Ry = np.array([[cosval[1], 0, sinval[1]], [0, 1, 0], [-sinval[1], 0, cosval[1]]])
    Rz = np.array([[cosval[2], -sinval[2], 0], [sinval[2], cosval[2], 0], [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    pc = np.dot(pc, np.transpose(R))
    return pc

def random_rotate_z(pc, angle=None):
    # Randomly rotate around z-axis
    if angle is None:
        angle = np.random.uniform() * 2 * np.pi
    cosval, sinval = np.cos(angle), np.sin(angle)
    R = np.array([[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]])
    return np.dot(pc, np.transpose(R))

def random_rotate_xyz(pc):
    # Randomly rotate around x, y, z axis
    angles = np.random.uniform(size=[3]) * 2 * np.pi
    cosval, sinval = np.cos(angles), np.sin(angles)
    Rx = np.array([[1, 0, 0], [0, cosval[0], -sinval[0]], [0, sinval[0], cosval[0]]])
    Ry = np.array([[cosval[1], 0, sinval[1]], [0, 1, 0], [-sinval[1], 0, cosval[1]]])
    Rz = np.array([[cosval[2], -sinval[2], 0], [sinval[2], cosval[2], 0], [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    pc = np.dot(pc, np.transpose(R))
    return pc

def augment_pc(pc):
    pc = random_scale_pc(pc)
    pc = shift_pc(pc)
    # pc = rotate_perturbation_pc(pc)
    pc = random_rotate_z(pc)
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point
