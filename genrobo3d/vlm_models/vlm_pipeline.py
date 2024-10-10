import os
import sys
import time
import json
import numpy as np
from easydict import EasyDict
import open3d as o3d

from PIL import Image
from PIL import ImageDraw, ImageFont, ImageEnhance
import matplotlib.pyplot as plt

import collections
import copy
from sklearn.cluster import DBSCAN

import torch
import torch.nn.functional as F
from chamferdist import ChamferDistance

from .owlv2_detector import Owlv2ObjectDetector
from .sam_segmentor import SAMSegmentor

from .vlm_utils import (
    get_color_palette, draw_contour, weighted_average_embeds,
)
from genrobo3d.utils.point_cloud import voxelize_pcd
from .vlm_configs import (
    VLMRLBenchConfig, VLMRealConfig
)
from genrobo3d.utils.robot_box import RobotBox

COLOR_PALETTE = get_color_palette()


class ObjectInfo(EasyDict):
    keystep = 0
    view_ids = []
    obj_ids = []
    boxes = []
    masks = []
    image_class_embeds = None
    objectness_scores = None
    pcd_xyz = None
    pcd_rgb = None
    captions = []
    caption_3d = None
    
    
class VLMPipeline(object):
    def __init__(
            self, det_model_id='large', sam_model_id='large',
            use_2d_caption=False, use_3d_caption=False,
            env_name='rlbench',
        ):
        """
        Load dection, segmentation, and caption models.
        Neet 2 A100 gpus to load all the models (3d captioner).
        """
        self.env_name = env_name
        if self.env_name == 'rlbench':
            self.vlm_config = VLMRLBenchConfig()
        elif self.env_name == 'real':
            self.vlm_config = VLMRealConfig()
        else:
            raise NotImplementedError(f'unsupported env_name: {env_name}')

        self.device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        start_time = time.time()
        self.det_model = Owlv2ObjectDetector(model_id=det_model_id, device=device)
        print('load detector: %.2fmin' % ((time.time() - start_time)/60.))

        start_time = time.time()
        self.sam_model = SAMSegmentor(model_id=sam_model_id, device=device)
        print('load segmentor: %.2fmin' % ((time.time() - start_time)/60.))

        self.caption_2d_model = None
        self.caption_3d_model = None

        # my own modification of https://github.com/krrish94/chamferdist/tree/master
        self.chamfer_dist_fn = ChamferDistance()

        self.reset_cache()

    def reset_cache(self):
        self.cache = EasyDict()
        self.robot_box = None

    def run(
            self, rgb_images, pcd_images, arm_links_info, debug=False
        ):
        """
        Args:
            rgb_images: (num_cameras, height, width, 3)
            pcd_images: (num_cameras, height, width, 3)
            arm_links_info: (bbox_info_dict, pose_info_dict)
        Returns:
            cache: dict of all the intermediate results
        """
        self.reset_cache()

        self.robot_box = RobotBox(arm_links_info, env_name=self.env_name)

        num_images = len(rgb_images)
        image_height, image_width = rgb_images.shape[1:3]

        # Run object detection
        # image_embeds torch.Size([num_images, 60, 60, 768])
        # pred_boxes torch.Size([num_images, 3600, 4])
        # objectness_logits torch.Size([num_images, 3600])
        # image_class_embeds torch.Size([num_images, 3600, 512])
        # class_logit_shift torch.Size([num_images, 3600, 1])
        # class_logit_scale torch.Size([num_images, 3600, 1])
        det_image_outputs = self.det_model.encode_images(rgb_images)
        self.cache.det_image_outputs = det_image_outputs

        # List of {"scores": (num_boxes, ), "boxes": (num_boxes, 4), "patch_indexs": (num_boxes, ), "patch_coords": (num_boxes, )}
        det_results = self.det_model.post_process_objectness_detection(
            det_image_outputs, **self.vlm_config.det_postprocess
        )
        self.cache.det_results = det_results

        # Run segmentation for each detected bounding boxes
        box_resize = max(image_width, image_height)
        input_boxes = [(det_results[k]['boxes'].numpy() * box_resize).tolist() for k in range(num_images)]
        # input_points = [(det_results[k]['patch_coords'].numpy() * image_size).tolist() for k in range(num_images)]

        # List of {"scores": (num_boxes, 3), "masks": (num_boxes, 3, height, width)}
        sam_results = self.sam_model(rgb_images, input_boxes)#, input_points)
        self.cache.sam_results = sam_results

        # Run captioning model
        if self.caption_2d_model is not None:
            som_images, som_num_objects = self.prepare_som_images(
                rgb_images, sam_results, use_contour=False
            )
            for i, som_image in enumerate(som_images):
                if som_image is not None:
                    if debug:
                        plt.imshow(som_image)
                        plt.show()
                    listed_captions = self.caption_2d_model(
                        som_image, num_objects=som_num_objects[i], debug=debug
                    )
                    sam_results[i]['captions'] = np.array(listed_captions)
            self.cache.sam_results = sam_results

        # Clean detected bboxes according to segmentation and masks
        cleaned_det_results, cleaned_sam_results = self.clean_det_bboxes(
            det_results, sam_results, pcd_images, self.robot_box, debug=debug
        )
        self.cache.cleaned_det_results = cleaned_det_results
        self.cache.cleaned_sam_results = cleaned_sam_results

        objects = self.merge_multiview_objects(
            det_image_outputs, cleaned_det_results, cleaned_sam_results, 
            rgb_images, pcd_images, self.robot_box, debug=debug
        )
        self.cache.objects = objects

        # Generate 3d captions
        if self.caption_3d_model is not None:
            objects = self.generate_3d_captions(objects)
            self.cache.objects = objects

        return self.cache

    def ground_object_with_query(self, text, objects=None, debug=False, return_sims=False):
        """
        Args:
            text: a string
        Returns:
            (id, obj) in objects list
        """
        if objects is None:
            assert 'objects' in self.cache
            objects = self.cache['objects']

        # (dim, )
        query_embeds = self.det_model.encode_texts([text]).text_embeds[0]
        query_embeds = query_embeds / (torch.linalg.norm(query_embeds, dim=-1, keepdim=True) + 1e-6)

        it_sims = []
        for obj in objects:
            if obj.image_class_embeds is None:  # obstacle
                # assert isinstance(obj.captions, list) and obj.captions[0] == 'obstacle'
                continue
            # image_class_embeds = obj.image_class_embeds
            image_class_embeds = weighted_average_embeds(
                obj.image_class_embeds, obj.objectness_scores, keepdim=False
            )
            image_class_embeds = image_class_embeds / (torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-6)
            it_sims.append(
                (query_embeds * image_class_embeds).sum().item()
            )

        # # TODO: for some tasks there is only 1 object in the scene
        # if 'drawer' in text or 'laptop' in text or 'shelf' in text or 'safe' in text \
        #     or 'microwave' in text or 'door' in text or 'box' in text:
        #     it_sims = [len(obj.pcd_xyz) for obj in objects if obj.image_class_embeds is not None]

        if len(it_sims) == 0:
            best_obj_id = None
            best_obj = None
        else:
            best_obj_id = np.argmax(it_sims)
            best_obj = objects[best_obj_id]

        if debug:
            print('all sims', it_sims)
            print('best obj_id and sim', best_obj_id)

        if return_sims:
            return best_obj_id, best_obj, it_sims
        return best_obj_id, best_obj

    def classify_objects_with_queries(self, texts, objects=None, add_robot_obtacle=True, debug=False):
        """
        Args:
            texts: a list of string
        Returns:
            labels: a list, text label for each object
        """
        if objects is None:
            assert 'objects' in self.cache
            objects = self.cache['objects']

        # (num_texts, dim)
        query_embeds = self.det_model.encode_texts(texts).text_embeds
        query_embeds = query_embeds / (torch.linalg.norm(query_embeds, dim=-1, keepdim=True) + 1e-6)

        pred_labels = []
        for k, obj in enumerate(objects):
            if len(obj.captions) > 0 and obj.captions[0] in ['robot', 'obstacle']:  # obstacle/robot
                if add_robot_obtacle:
                    pred_labels.append(obj.captions[0])
                continue
            image_class_embeds = weighted_average_embeds(
                obj.image_class_embeds, obj.objectness_scores, keepdim=False
            )
            # image_class_embeds = obj.image_class_embeds
            image_class_embeds = image_class_embeds / (torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-6)
            it_sim = torch.einsum('d,td->t', image_class_embeds, query_embeds).data.cpu().numpy()
            # it_sim = np.max(torch.einsum('id,td->it', image_class_embeds, query_embeds).data.cpu().numpy(), 0)
            pred_label_id = np.argmax(it_sim)
            pred_labels.append(texts[pred_label_id])
            if debug:
                print(f'obj={k}, all sims {it_sim}, best_label {pred_label_id}={pred_labels[-1]}')
        return pred_labels

    def clean_det_bboxes(
        self, det_results, sam_results, pcd_images, robot_box, debug=False
    ):
        """
        Clean the detection bounding boxes:
            1. Remove bboxes that contain much background (with predefined workspace)
            2. Remove bboxes that mainly capture robot arms (with predefined robot boxes)
            3. Remove bboxes that mainly capture tables (with predefined table height)
        """
        new_det_results, new_sam_results = [], []

        for view_id, (det_res, sam_res, pcd_img) in enumerate(zip(det_results, sam_results, pcd_images)):

            valid_box_ids = []
            for k in range(len(det_res['boxes'])):
                obj_mask = sam_res['masks'][k][0]
                obj_pcd = pcd_img[obj_mask]

                # deal with real world noisy depth (most of the points are projected to a same value)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(obj_pcd)
                pcd = pcd.remove_duplicated_points()
                if self.env_name == 'real': # point cloud is very noisy
                    pcd.remove_statistical_outlier(
                        nb_neighbors=self.vlm_config.pcd_outlier_removal_config.nb_neighbors, 
                        std_ratio=self.vlm_config.pcd_outlier_removal_config.std_ratio
                    )
                obj_pcd = np.asarray(pcd.points)

                if debug:
                    print('view_id', view_id, 'obj_id', k)
                    plt.imshow(obj_mask)
                    plt.show()
                    print('pcd range', np.min(obj_pcd, 0), np.max(obj_pcd, 0))

                # Check if obj pcd is in workspace
                inws_mask = np.all(obj_pcd > self.vlm_config.workspace[0], -1) \
                            & np.all(obj_pcd < self.vlm_config.workspace[1], -1)
                if debug:
                    print('in workspace ratio', np.mean(inws_mask))
                if 1 - np.mean(inws_mask) > self.vlm_config.clean_det_config.max_out_workspace_ratio:
                    continue

                obj_pcd = obj_pcd[inws_mask]

                # Check if obj pcd overlaps much with robot arm & gripper
                robot_ratio = robot_box.get_pc_overlap_ratio(xyz=obj_pcd)
                if debug:
                    print('robot ratio', robot_ratio)
                if robot_ratio > self.vlm_config.clean_det_config.max_robot_ratio:
                    continue

                # Check if obj pcd overlaps much with the table
                table_ratio = np.mean(obj_pcd[:, 2] < self.vlm_config.table_height)
                # # TODO: Special case: target plane (small) on the table, e.g., 10cmx10cm
                # if table_ratio > self.vlm_config.clean_det_config.max_table_ratio and \
                #     np.prod(np.max(obj_pcd[..., :2], 0) - np.min(obj_pcd[..., :2], 0)) < 0.01:
                #     table_ratio = 0
                if debug:
                    print('table ratio', table_ratio)
                if table_ratio > self.vlm_config.clean_det_config.max_table_ratio:
                    continue

                if robot_ratio + table_ratio > 0.8:
                    continue

                valid_box_ids.append(k)
                if debug:
                    print(f'keep (view={view_id}, obj={k})\n')

            valid_box_ids = np.array(valid_box_ids)
            new_det_results.append({k: v[valid_box_ids] for k, v in det_res.items()})
            if len(valid_box_ids) == 0:
                new_sam_results.append(None)
            else:
                new_sam_results.append({k: v[valid_box_ids] for k, v in sam_res.items()})

        return new_det_results, new_sam_results

    def merge_multiview_objects(
        self, det_image_outputs, det_results, sam_results, 
        rgb_images, pcd_images, robot_box, debug=False
    ):
        image_height, image_width = rgb_images.shape[1:3]
        image_longest_edge = max(image_height, image_width)

        # Create all objects from multi-view images
        all_objects = []
        for view_id, (det_res, sam_res, rgb_img, pcd_img) in enumerate(zip(det_results, sam_results, rgb_images, pcd_images)):
            for k, (box, obj_score) in enumerate(zip(det_res['boxes'], det_res['scores'])):

                obj = ObjectInfo()
                obj.view_ids.append(view_id)
                obj.obj_ids.append(k)
                obj.boxes.append(box)
                obj.masks.append(sam_res['masks'][k][0])
                obj.objectness_scores = obj_score.unsqueeze(0)
                k_patch_index = det_res['patch_indexs'][k]
                obj.image_class_embeds = det_image_outputs.image_class_embeds[view_id][k_patch_index].unsqueeze(0)
                if 'captions' in sam_res:
                    obj.captions.append(sam_res['captions'][k])

                # Clean point clouds
                obj.pcd_xyz = pcd_img[sam_res['masks'][k][0]]
                # remove points outside of workspace, on/below the table, in the robot box
                cleaned_pcd_mask = self.clean_object_pcd(obj.pcd_xyz, robot_box)
                obj.pcd_xyz = obj.pcd_xyz[cleaned_pcd_mask]
                obj.pcd_rgb = rgb_img[sam_res['masks'][k][0]][cleaned_pcd_mask]

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(obj.pcd_xyz)
                pcd.colors = o3d.utility.Vector3dVector(obj.pcd_rgb)
                pcd = pcd.remove_duplicated_points()
                # downsampling
                pcd = pcd.voxel_down_sample(self.vlm_config.voxel_size)
                if self.env_name == 'real':
                    pcd.remove_statistical_outlier(
                        nb_neighbors=self.vlm_config.pcd_outlier_removal_config.nb_neighbors, 
                        std_ratio=self.vlm_config.pcd_outlier_removal_config.std_ratio
                    )
                obj.pcd_xyz = np.asarray(pcd.points).astype(np.float32)
                obj.pcd_rgb = np.asarray(pcd.colors).astype(np.uint8)

                # Separate bboxes that may contain more than one object by geometry distances
                clustering = DBSCAN(
                    eps=self.vlm_config.dbscan_config.eps, 
                    min_samples=self.vlm_config.dbscan_config.min_samples
                ).fit(obj.pcd_xyz)
                label_counter = collections.Counter(clustering.labels_)
                num_clusters = len([label for label in label_counter.keys() if label != -1])                    
                if debug:
                    print(f"view_id={view_id}, obj_id={k}, num_clusters={num_clusters}")
                    box = np.array([int(x*image_longest_edge) for x in box])
                    plt.imshow(rgb_img[max(box[1], 0):box[3], max(box[0], 0):box[2]])
                    plt.show()
                    print()
                # separate the object into parts
                if num_clusters > 1:
                    for label, npoints in label_counter.items():
                        if debug:
                            print('\tpart', label, npoints/len(obj.pcd_xyz))
                        if label != -1 and npoints / len(obj.pcd_xyz) > self.vlm_config.dbscan_config.min_keep_ratio:
                            obj_part = copy.deepcopy(obj)
                            part_mask = clustering.labels_ == label
                            obj_part.pcd_xyz = obj.pcd_xyz[part_mask]
                            obj_part.pcd_rgb = obj.pcd_rgb[part_mask]
                            if len(obj_part.pcd_xyz) > self.vlm_config.pcd_min_num_points:
                                all_objects.append(obj_part)
                else:
                    if len(obj.pcd_xyz) > self.vlm_config.pcd_min_num_points:
                        all_objects.append(obj)

        # Reorder all objects by number of points (descending order)
        all_objects.sort(key=lambda obj: -len(obj.pcd_xyz))

        # Merge all the remaining parts as obstacle
        obstacle = ObjectInfo()
        obstacle.pcd_xyz = np.empty((0, 3), dtype=np.float32)
        obstacle.pcd_rgb = np.empty((0, 3), dtype=np.float32)
        obstacle.captions = ['obstacle']
        for view_id, (det_res, sam_res, rgb_img, pcd_img) in enumerate(zip(det_results, sam_results, rgb_images, pcd_images)):
            obstacle_mask = np.ones(rgb_img.shape[:2], dtype=bool)
            for k, box in enumerate(det_res['boxes']):
                obstacle_mask[sam_res['masks'][k][0].numpy()] = False
            if np.sum(obstacle_mask) > 0:
                # obstacle.view_ids.append(view_id)
                # obstacle.masks.append(obstacle_mask)
                obstacle.pcd_xyz = np.concatenate([obstacle.pcd_xyz, pcd_img[obstacle_mask]], 0)
                obstacle.pcd_rgb = np.concatenate([obstacle.pcd_rgb, rgb_img[obstacle_mask]], 0)
        cleaned_pcd_mask = self.clean_object_pcd(obstacle.pcd_xyz, robot_box=None)
        obstacle.pcd_xyz = obstacle.pcd_xyz[cleaned_pcd_mask]
        obstacle.pcd_rgb = obstacle.pcd_rgb[cleaned_pcd_mask]
        obstacle.pcd_xyz, xyz_idxs = voxelize_pcd(obstacle.pcd_xyz, voxel_size=self.vlm_config.voxel_size)
        obstacle.pcd_rgb = obstacle.pcd_rgb[xyz_idxs]
        if self.env_name == 'real':
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(obstacle.pcd_xyz)
            pcd.colors = o3d.utility.Vector3dVector(obstacle.pcd_rgb)
            pcd = pcd.remove_duplicated_points()
            pcd.remove_statistical_outlier(
                nb_neighbors=self.vlm_config.pcd_outlier_removal_config.nb_neighbors, 
                std_ratio=self.vlm_config.pcd_outlier_removal_config.std_ratio
            )
            obstacle.pcd_xyz = np.asarray(pcd.points).astype(np.float32)
            obstacle.pcd_rgb = np.asarray(pcd.colors).astype(np.uint8)

        # Separate robot from objects
        robot = ObjectInfo()
        robot.captions = ['robot']
        robot_point_idxs = np.array(list(
            robot_box.get_pc_overlap_ratio(xyz=obstacle.pcd_xyz, return_indices=True)[1]
        ))
        if len(robot_point_idxs) > 0:
            robot.pcd_xyz = obstacle.pcd_xyz[robot_point_idxs]
            robot.pcd_rgb = obstacle.pcd_rgb[robot_point_idxs]
            robot.pcd_xyz, xyz_idxs = voxelize_pcd(robot.pcd_xyz, voxel_size=self.vlm_config.voxel_size)
            robot.pcd_rgb = robot.pcd_rgb[xyz_idxs]

        # Remaining is the obstacle
        obstacle_mask = np.ones(obstacle.pcd_xyz.shape[0], dtype=bool)
        if len(robot_point_idxs) > 0:
            obstacle_mask[robot_point_idxs] = False
        obstacle.pcd_xyz = obstacle.pcd_xyz[obstacle_mask]
        obstacle.pcd_rgb = obstacle.pcd_rgb[obstacle_mask]

        merged_objects = []

        if len(all_objects) > 0:
            # Remove points in obstacle that overlap with objects
            if self.env_name == 'rlbench':
                # obstacle can also be separated into parts and merge these with the other objects?
                clustering = DBSCAN(
                    eps=self.vlm_config.dbscan_config.eps, 
                    min_samples=self.vlm_config.dbscan_config.min_samples
                ).fit(obstacle.pcd_xyz)
                label_counter = collections.Counter(clustering.labels_)
                num_clusters = len([label for label in label_counter.keys() if label != -1])                    
                # separate the object into parts
                obstacle_mask = np.ones((obstacle.pcd_xyz.shape[0], ), dtype=bool)
                for label, npoints in label_counter.items():
                    if debug:
                        print('part', label, npoints/len(obstacle.pcd_xyz))
                    if label != -1:
                        part_mask = clustering.labels_ == label
                        if np.mean(part_mask) < 0.1:
                            continue
                        src_obj_pcd = torch.from_numpy(obstacle.pcd_xyz[part_mask]).to(self.device).unsqueeze(0)
                        # Find the best match in previous objects
                        best_dist, best_idx = np.inf, None
                        for idx, obj in enumerate(all_objects):
                            tgt_obj_pcd = torch.from_numpy(obj.pcd_xyz).to(self.device).unsqueeze(0)
                            src2tgt_dist = np.sqrt(self.chamfer_dist_fn(src_obj_pcd, tgt_obj_pcd, point_reduction=self.vlm_config.merge_obj_config.chamfer_dist_measure)[0].item())
                            tgt2src_dist = np.sqrt(self.chamfer_dist_fn(tgt_obj_pcd, src_obj_pcd, point_reduction=self.vlm_config.merge_obj_config.chamfer_dist_measure)[0].item())
                            pcd_dist = min(src2tgt_dist, tgt2src_dist)
                            if pcd_dist < best_dist:
                                best_idx = idx
                                best_dist = pcd_dist
                        if debug:
                            print('\tbest', 'id', best_idx, 'dist', best_dist)
                        if best_dist < self.vlm_config.merge_obj_config.max_match_pcd_dist:
                            all_objects[best_idx].pcd_xyz = np.concatenate(
                                [all_objects[best_idx].pcd_xyz, obstacle.pcd_xyz[part_mask]], 0
                            )
                            all_objects[best_idx].pcd_rgb = np.concatenate(
                                [all_objects[best_idx].pcd_rgb, obstacle.pcd_rgb[part_mask]], 0
                            )
                            obstacle_mask[part_mask] = False
                        
                obstacle.pcd_xyz = obstacle.pcd_xyz[obstacle_mask]
                obstacle.pcd_rgb = obstacle.pcd_rgb[obstacle_mask]

            # Merge smaller objects to larger ones
            all_objects.sort(key=lambda obj: -len(obj.pcd_xyz))

            merged_objects.append(all_objects[0])
            for obj in all_objects[1:]:
                best_matched_obj = None
                if debug:
                    print(f'cand (view_id={obj.view_ids[0]}, obj_id={obj.obj_ids[0]})')
                for eid, exist_obj in enumerate(merged_objects):
                    # if (self.env_name == 'real' and obj.view_ids[0] not in exist_obj.view_ids) or (self.env_name == 'rlbench'):
                    # if True:
                    # Do not merge bboxes of the same view
                    if obj.view_ids[0] not in exist_obj.view_ids:
                        src_obj_pcd = torch.from_numpy(obj.pcd_xyz).to(self.device).unsqueeze(0)
                        tgt_obj_pcd = torch.from_numpy(exist_obj.pcd_xyz).to(self.device).unsqueeze(0)
                        src2tgt_dist = np.sqrt(self.chamfer_dist_fn(src_obj_pcd, tgt_obj_pcd, point_reduction=self.vlm_config.merge_obj_config.chamfer_dist_measure)[0].item())
                        tgt2src_dist = np.sqrt(self.chamfer_dist_fn(tgt_obj_pcd, src_obj_pcd, point_reduction=self.vlm_config.merge_obj_config.chamfer_dist_measure)[0].item())
                        pcd_dist = min(src2tgt_dist, tgt2src_dist)

                        normed_ft1 = F.normalize(
                            weighted_average_embeds(exist_obj.image_class_embeds, exist_obj.objectness_scores, keepdim=True),
                            p=2, dim=-1
                        )[0]
                        normed_ft2 = F.normalize(obj.image_class_embeds[None, :], p=2, dim=-1)[0]
                        embed_sim = (normed_ft1 * normed_ft2).sum().item()
                        if debug:
                            print(
                                f'\t{eid}: (view_id={exist_obj.view_ids[0]}, obj_id={exist_obj.obj_ids[0]})',  
                                'src2tgt_dist', src2tgt_dist, 'tgt2src_dist', tgt2src_dist, 'embed_sim', embed_sim
                            )
                        if best_matched_obj is None:
                            best_matched_obj = (eid, pcd_dist, embed_sim)
                        else:
                            if self.env_name == 'rlbench':
                                # TODO: weight pcd_dist more
                                if embed_sim/max(pcd_dist, 0.005) > best_matched_obj[2]/max(best_matched_obj[1], 0.005):
                                    best_matched_obj = (eid, pcd_dist, embed_sim)
                            else:
                                # weight embed_sim more
                                if embed_sim/max(pcd_dist, 0.01) > best_matched_obj[2]/max(best_matched_obj[1], 0.01):
                                    best_matched_obj = (eid, pcd_dist, embed_sim)
                
                # If two objects are very close, directly merge them
                if best_matched_obj is not None and \
                    ((best_matched_obj[2] > self.vlm_config.merge_obj_config.min_match_embed_sim \
                      and best_matched_obj[1] < self.vlm_config.merge_obj_config.max_match_pcd_dist)\
                    or (self.env_name == 'rlbench' and best_matched_obj[1] < 0.01)):
                    
                    exist_obj = merged_objects[best_matched_obj[0]]
                    exist_obj.view_ids.extend(obj.view_ids)
                    exist_obj.obj_ids.extend(obj.obj_ids)
                    exist_obj.boxes.extend(obj.boxes)
                    exist_obj.masks.extend(obj.masks)
                    exist_obj.captions.extend(obj.captions)
                    exist_obj.pcd_xyz = np.concatenate([exist_obj.pcd_xyz, obj.pcd_xyz], 0)
                    exist_obj.pcd_rgb = np.concatenate([exist_obj.pcd_rgb, obj.pcd_rgb], 0)

                    if debug:
                        print(f'\tmerged to obj {best_matched_obj[0]}, (view_id={exist_obj.view_ids[0]}, obj_id={exist_obj.obj_ids[0]})')

                    exist_obj.pcd_xyz, xyz_idxs = voxelize_pcd(exist_obj.pcd_xyz, self.vlm_config.voxel_size)
                    exist_obj.pcd_rgb = exist_obj.pcd_rgb[xyz_idxs]

                    exist_obj.image_class_embeds = torch.cat([exist_obj.image_class_embeds, obj.image_class_embeds], 0)
                    exist_obj.objectness_scores = torch.cat([exist_obj.objectness_scores, obj.objectness_scores], 0)
                    # exist_obj.image_class_embeds = (exist_obj.image_class_embeds * n_embeds + obj.image_class_embeds) / (n_embeds + 1)
                else:
                    merged_objects.append(obj)

        if robot.pcd_xyz is not None and len(robot.pcd_xyz) > self.vlm_config.pcd_min_num_points:
            merged_objects.append(robot)
        if len(obstacle.pcd_xyz) > self.vlm_config.pcd_min_num_points:
            merged_objects.append(obstacle)

        # Postprocess all the point clouds
        remained_objects = []
        for obj in merged_objects:
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(obj.pcd_xyz)
            # pcd.colors = o3d.utility.Vector3dVector(obj.pcd_rgb)
            # # downsampling
            # pcd = pcd.voxel_down_sample(self.vlm_config.voxel_size)
            # # remove outliers that have few neighbors in a given sphere
            # pcd, _ = pcd.remove_radius_outlier(nb_points=16, radius=0.03)
            # # print('#points', len(pcd.points))
            # obj.pcd_xyz = np.asarray(pcd.points).astype(np.float32)
            # obj.pcd_rgb = np.asarray(pcd.colors).astype(np.uint8)
            if len(obj.pcd_xyz) > self.vlm_config.pcd_min_num_points:
                remained_objects.append(obj)
        merged_objects = remained_objects

        return merged_objects

    def prepare_som_images(self, rgb_images, sam_results, use_contour=False):
        """
        Add numeric markers on each segmentation mask
        Returns:
            som_images: list of PIL.Image
            num_objects: list of int
        """
        som_images, num_objects = [], []

        for i, (sam_res, rgb_img) in enumerate(zip(sam_results, rgb_images)):
            rgb_img = np.copy(rgb_img)

            if sam_res is None:
                som_images.append(None)
            else:
                num_objects.append(len(sam_res['masks']))

                final_image = Image.new('RGBA', rgb_img.shape[:2], (0, 0, 0, 0))
                for k, (s, m) in enumerate(zip(sam_res['scores'], sam_res['masks'])):
                    m = m[0].numpy()

                    # TODO: this is not good if the mask is noisy
                    if use_contour:
                        rgb_img = draw_contour(rgb_img, m, COLOR_PALETTE[np.random.randint(len(COLOR_PALETTE))])
                        # plt.imshow(m[0])
                        # plt.show()

                    xsum = np.sum(m, 0)
                    ysum = np.sum(m, 1)
                    x = np.median(np.nonzero(xsum)).astype(np.int32)
                    y = np.median(np.nonzero(ysum)).astype(np.int32)
                    # print(m[0][y, x], x, y, xsum[x], ysum[y])
                    if not m[y, x].item():  # the center is not inside the mask
                        if xsum[x] > ysum[y]:
                            y = np.nonzero(m[:, x])
                            y = np.median(y).astype(np.int32)
                        else:
                            x = np.nonzero(m[y, :])
                            x = np.median(x).astype(np.int32)
                        # print('new x, y', x, y)
                    mask_image = Image.new(mode='RGB', size=(10, 10), color='black')

                    draw = ImageDraw.ImageDraw(mask_image)
                    font_file = os.path.join(os.path.dirname(__file__), "../../assets/arial.ttf")
                    font = ImageFont.truetype(font_file, 8)
                    draw.text((1, 1), str(k+1), fill="white", font=font)

                    mask_image = mask_image.convert('RGBA')
                    r, g, b, a = mask_image.split()
                    opacity = 0.8
                    alpha = ImageEnhance.Brightness(a).enhance(opacity)
                    mask_image.putalpha(alpha)

                    final_image.paste(mask_image, (x-7, y-7))
                
                rgb_img = Image.fromarray(rgb_img)
                final_image = Image.composite(final_image, rgb_img, final_image)
                final_image = final_image.convert('RGB')
                # final_image.show()
                # final_image.save('view%d.png'%i)
                som_images.append(final_image)
        return som_images, num_objects

    def generate_3d_captions(self, objects):
        for obj in objects:
            if obj.captions[0] != 'obstacle' and obj.captions[0] != 'robot':
                caption_3d = self.caption_3d_model(obj.pcd_xyz, obj.captions)
                obj.caption_3d = caption_3d
        return objects
    
    def clean_object_pcd(self, pcd_xyz, robot_box=None):
        """
        Return remaining pcd_masks
        """
        # remove points outside of workspace
        pcd_masks = np.all(pcd_xyz > self.vlm_config.workspace[0], -1) \
                    & np.all(pcd_xyz < self.vlm_config.workspace[1], -1)

        # remove points inside robot_box
        if robot_box is not None:
            _, overlap_indices = robot_box.get_pc_overlap_ratio(xyz=pcd_xyz, return_indices=True)
            if len(overlap_indices) > 0:
                pcd_masks[np.array(list(overlap_indices))] = False

        # remove points close to table
        # # TODO: special case: target plane (small) on the table, e.g., 10cmx10cm
        pcd_masks[pcd_xyz[..., 2] < self.vlm_config.table_height] = False
        # table_ratio = np.mean(table_dist < self.vlm_config.table_dist_threshold)
        # if table_ratio > self.vlm_config.clean_det_config.max_table_ratio and \
        #     np.prod(np.max(pcd_xyz[..., :2], 0) - np.min(pcd_xyz[..., :2], 0)) < 0.01:
        #     pass
        # else:
        #     pcd_masks[table_dist < self.vlm_config.table_dist_threshold] = False

        return pcd_masks
