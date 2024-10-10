import numpy as np
import einops
import collections
import time


def get_disc_gt_pos_prob(
    xyz, gt_pos, pos_bin_size=0.01, pos_bins=50, heatmap_type='plain', robot_point_idxs=None
):
    '''
    heatmap_type:
        - plain: the same prob for all voxels with distance to gt_pos within pos_bin_size
        - dist: prob for each voxel is propotional to its distance to gt_pos
    '''
    shift = np.arange(-pos_bins, pos_bins) * pos_bin_size # (pos_bins*2, )
    cands_pos = np.stack([shift] * 3, 0)[None, :, :] + xyz[:, :, None] # (npoints, 3, pos_bins*2)
    dists = np.abs(gt_pos[None, :, None] - cands_pos) # (npoints, 3, pos_bins*2)
    dists = einops.rearrange(dists, 'n c b -> c (n b)') # (3, npoints*pos_bins*2)
    
    if heatmap_type == 'plain':
        disc_pos_prob = np.zeros((3, xyz.shape[0] * pos_bins * 2), dtype=np.float32)
        disc_pos_prob[dists < 0.01] = 1
        if robot_point_idxs is not None and len(robot_point_idxs) > 0:
            disc_pos_prob = einops.rearrange(disc_pos_prob, 'c (n b) -> c n b', n=xyz.shape[0])
            disc_pos_prob[:, robot_point_idxs] = 0
            disc_pos_prob = einops.rearrange(disc_pos_prob, 'c n b -> c (n b)')
        for i in range(3):
            if np.sum(disc_pos_prob[i]) == 0:
                disc_pos_prob[i, np.argmin(dists[i])] = 1
        disc_pos_prob = disc_pos_prob / np.sum(disc_pos_prob, -1, keepdims=True)
        # disc_pos_prob = einops.rearrange(disc_pos_prob, 'c (n b) -> c n b')
    else:
        disc_pos_prob = 1 / np.maximum(dists, 1e-4)
        # TODO
        # disc_pos_prob[dists > 0.02] = 0
        disc_pos_prob[dists > 0.01] = 0
        if robot_point_idxs is not None and len(robot_point_idxs) > 0:
            disc_pos_prob = einops.rearrange(disc_pos_prob, 'c (n b) -> c n b', n=xyz.shape[0])
            disc_pos_prob[:, robot_point_idxs] = 0
            disc_pos_prob = einops.rearrange(disc_pos_prob, 'c n b -> c (n b)')
        for i in range(3):
            if np.sum(disc_pos_prob[i]) == 0:
                disc_pos_prob[i, np.argmin(dists[i])] = 1
        disc_pos_prob = disc_pos_prob / np.sum(disc_pos_prob, -1, keepdims=True)
    
    return disc_pos_prob

def get_best_pos_from_disc_pos(disc_pos_prob, xyz, pos_bin_size=0.01, pos_bins=50, best='max', topk=1000):
    '''Args:
        disc_pos_prob: (3, npoints*pos_bins*2)
        xyz: (npoints, 3)
    '''
    assert best in ['max', 'ens1']
    shift = np.arange(-pos_bins, pos_bins) * pos_bin_size # (pos_bins*2, )
    cands_pos = np.stack([shift] * 3, 0)[None, :, :] + xyz[:, :, None] # (npoints, 3, pos_bins*2)
    
    if best == 'max':
        cands_pos = einops.rearrange(cands_pos, 'n c b -> c (n b)') # (3, npoints*pos_bins*2)
        idxs = np.argmax(disc_pos_prob, -1)
        best_pos = cands_pos[np.arange(3), idxs]
        
    elif best == 'ens1':
        # st = time.time()
        cands_pos = einops.rearrange(cands_pos, 'n c b -> c (n b)') # (3, npoints*pos_bins*2)
        # disc_pos_prob = torch.from_numpy(disc_pos_prob)
        # disc_pos_prob = torch.softmax(disc_pos_prob, -1).numpy()
        cands_pos_voxel = np.round(cands_pos / 0.005).astype(np.int32) # (3, npoints*pos_bins*2)
        idxs = np.argsort(-disc_pos_prob, -1)#[:, :topk]
        best_pos = []
        for i in range(3):
            sparse_values = collections.defaultdict(int)
            # for k in idxs[i, :topk]:
            for k in idxs[i]:
                sparse_values[cands_pos_voxel[i, k].item()] += disc_pos_prob[i, k]
            best_pos_i, best_value = None, -np.inf
            for k, v in sparse_values.items():
                if v > best_value:
                    best_value = v
                    best_pos_i = k
            best_pos.append(best_pos_i * 0.005)
            # print(i, 'npoints', xyz.shape, 'uniq voxel', len(sparse_values), best_value)
        best_pos = np.array(best_pos)
        # print('time', time.time() - st)
        
    # else:
    #     # disc_pos_prob = torch.from_numpy(disc_pos_prob)
    #     # disc_pos_logprob = torch.log_softmax(disc_pos_prob.transpose(0, 1).reshape(3, -1), -1).reshape(3, -1, 100).transpose(0, 1).numpy()
    #     # disc_pos_prob = disc_pos_prob.numpy()
        
    #     disc_pos_prob = einops.rearrange(disc_pos_prob, 'c (n b) -> n c b', b=pos_bins*2)
    #     cands_pos_voxel = np.round(cands_pos / 0.005).astype(np.int32) # (npoints, 3, pos_bins*2)
    #     sparse_values = collections.defaultdict(int)
    #     for i in range(xyz.shape[0]):
    #         idxs = np.argsort(-disc_pos_prob[i], -1)
    #         for kx in idxs[0, :topk]:
    #             for ky in idxs[1, :topk]:
    #                 for kz in idxs[2, :topk]:
    #                     sparse_values[(cands_pos_voxel[i, 0, kx], cands_pos_voxel[i, 1, ky], cands_pos_voxel[i, 2, kz])] += \
    #                         disc_pos_prob[i, 0, kx] * disc_pos_prob[i, 1, ky] * disc_pos_prob[i, 2, kz]

    #     # cands_pos = einops.rearrange(cands_pos, 'n c b -> c (n b)') # (3, npoints*pos_bins*2)
    #     # cands_pos_voxel = np.round(cands_pos / 0.005).astype(np.int32)
    #     # sparse_values = collections.defaultdict(int)
    #     # idxs = np.argsort(-disc_pos_prob[0], -1)
    #     # for k in idxs[:topk]:
    #     #     sparse_values[(cands_pos_voxel[0, k], cands_pos_voxel[1, k], cands_pos_voxel[2, k])] += \
    #     #                     disc_pos_prob[0, k] * disc_pos_prob[1, k] * disc_pos_prob[2, k]
    #     # best_pos, best_prob = None, -np.inf
    #     # for k, v in sparse_values.items():
    #     #     if v > best_prob:
    #     #         best_prob = v
    #     #         best_pos = k
    #     best_pos = np.array(best_pos).astype(np.float32) * 0.005
    return best_pos
