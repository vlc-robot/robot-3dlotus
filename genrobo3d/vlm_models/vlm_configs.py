from easydict import EasyDict
import numpy as np

from genrobo3d.configs.rlbench.constants import get_robot_workspace

class VLMRLBenchConfig(object):
    robot_workspace = get_robot_workspace(real_robot=False)
    workspace = np.array(
        [[robot_workspace['X_BBOX'][0], robot_workspace['Y_BBOX'][0], robot_workspace['Z_BBOX'][0]], 
         [robot_workspace['X_BBOX'][1], robot_workspace['Y_BBOX'][1], robot_workspace['Z_BBOX'][1]]]
    )
    table_height = robot_workspace['TABLE_HEIGHT']

    # point cloud downsampling
    voxel_size = 0.01  # 1cm

    # detection: object bbox postprocess
    det_postprocess = EasyDict({
        "threshold": 0.1,            # objectness score theshold
        "target_sizes": None,        # None: bbox coords are ratio, otherwise times (w, h)
        "min_size_ratio": None,      # remove objects that are too small
        "max_size_ratio": 0.8,       # remove objects that are too large
        "min_return_topk": 1,       # return at least min objects
        "max_return_topk": 10,      # return at most max objects
        "use_nms": True,
        "nms_sigma": 0.2,
        "nms_thresh": 0.1,
    })

    # clean detection bboxes
    table_dist_threshold = 0.0025
    clean_det_config = EasyDict(
        max_out_workspace_ratio = 0.2,
        max_robot_ratio = 0.5, 
        max_table_ratio = 0.5,
    )

    # merge object
    merge_obj_config = EasyDict(
        chamfer_dist_measure = 'min',
        max_match_pcd_dist = 0.02,
        min_match_embed_sim = 0.6,
    )
    
    # dbscan config
    dbscan_config = EasyDict(
        eps = 0.02, 
        min_samples = 5,
        min_keep_ratio = 0.3,
    )

    pcd_min_num_points = 20

class VLMRealConfig(object):
    robot_workspace = get_robot_workspace(real_robot=True, use_vlm=True)

    workspace = np.array(
        [[robot_workspace['X_BBOX'][0], robot_workspace['Y_BBOX'][0], robot_workspace['Z_BBOX'][0]], 
         [robot_workspace['X_BBOX'][1], robot_workspace['Y_BBOX'][1], robot_workspace['Z_BBOX'][1]]]
    )
    table_height = robot_workspace['TABLE_HEIGHT']

    # point cloud downsampling
    voxel_size = 0.01  # 1cm

    # detection: object bbox postprocess
    det_postprocess = EasyDict({
        "threshold": 0.15,            # objectness score theshold
        "target_sizes": None,        # None: bbox coords are ratio, otherwise times (w, h)
        "min_size_ratio": None,      # remove objects that are too small
        "max_size_ratio": 0.8,       # remove objects that are too large
        "min_return_topk": 1,       # return at least min objects
        "max_return_topk": 10,      # return at most max objects
        "use_nms": True,
        "nms_sigma": 0.2,
        "nms_thresh": 0.1,
    })

    # clean detection bboxes
    clean_det_config = EasyDict(
        max_out_workspace_ratio = 0.35,
        max_robot_ratio = 0.5, 
        max_table_ratio = 0.75,
    )

    # merge object
    merge_obj_config = EasyDict(
        chamfer_dist_measure = 'min',
        max_match_pcd_dist = 0.1,
        min_match_embed_sim = 0.8,
    )
    
    # pcd outlier removal
    pcd_outlier_removal_config = EasyDict(
        nb_neighbors = 50,
        std_ratio = 0.2
    )

    # dbscan config
    dbscan_config = EasyDict(
        eps = 0.015, 
        min_samples = 5,
        min_keep_ratio = 0.4,
    )

    pcd_min_num_points = 20

