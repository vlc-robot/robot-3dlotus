import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d

class RobotBox(object):

    def __init__(self, arm_links_info, env_name='rlbench', keep_gripper=False):
        bbox_info, pose_info = arm_links_info

        self.robot_obboxes = []

        if env_name == 'rlbench':
            arm_links = ["Panda_link0", "Panda_link1", "Panda_link2", "Panda_link3", "Panda_link4", "Panda_link5", "Panda_link6", "Panda_link7"]
            if not keep_gripper:
                arm_links.extend(["Panda_rightfinger", "Panda_leftfinger", "Panda_gripper"])

            for arm_link in arm_links:
                if arm_link in ["Panda_link0", "Panda_rightfinger", "Panda_leftfinger", "Panda_gripper"]:
                    link_bbox = bbox_info[f"{arm_link}_visual_bbox"]
                    link_pose = pose_info[f"{arm_link}_visual_pose"]
                else:
                    link_bbox = bbox_info[f"{arm_link}_respondable_bbox"]
                    link_pose = pose_info[f"{arm_link}_respondable_pose"]

                link_rot = R.from_quat(link_pose[3:]).as_matrix()   # xyzw
                obbox = o3d.geometry.OrientedBoundingBox(
                    link_pose[:3], link_rot, link_bbox[1::2] - link_bbox[::2]
                )

                self.robot_obboxes.append(obbox)

        elif env_name == 'real':
            rm_links = [
                'left_base_link_bbox', 'left_shoulder_link_bbox', 'left_upper_arm_link_bbox', 
                'left_forearm_link_bbox', 'left_wrist_1_link_bbox', 'left_wrist_2_link_bbox', 
                'left_wrist_3_link_bbox', 'left_ft300_mounting_plate_bbox', 'left_ft300_sensor_bbox'
            ]
            if not keep_gripper:
                rm_links.extend([
                    'left_camera_link_bbox', 'left_gripper_body_bbox', 'left_gripper_bracket_bbox', 'left_gripper_finger_1_finger_tip_bbox', 'left_gripper_finger_1_flex_finger_bbox', 'left_gripper_finger_1_safety_shield_bbox', 'left_gripper_finger_1_truss_arm_bbox', 'left_gripper_finger_1_moment_arm_bbox', 'left_gripper_finger_2_finger_tip_bbox', 'left_gripper_finger_2_flex_finger_bbox', 'left_gripper_finger_2_safety_shield_bbox', 'left_gripper_finger_2_truss_arm_bbox', 'left_gripper_finger_2_moment_arm_bbox'
                ])
            rm_links = set(rm_links)
            for arm_link, link_bbox in bbox_info.items():
                if arm_link in rm_links:
                    link_pose = pose_info[arm_link.replace('_bbox', '_pose')]
                    link_rot = R.from_quat(link_pose[3:]).as_matrix()   # xyzw
                    obbox = o3d.geometry.OrientedBoundingBox(
                        link_pose[:3], link_rot, np.array(link_bbox[1::2]) - np.array(link_bbox[::2])
                    )
                    self.robot_obboxes.append(obbox)

    def get_pc_overlap_ratio(self, xyz=None, pcd=None, return_indices=False):
        if xyz is None:
            assert pcd is not None
            points = pcd.points
        else:
            points = o3d.utility.Vector3dVector(xyz)

        num_points = max(len(points), 1)
        overlap_point_ids = set()
        for obbox in self.robot_obboxes:
            tmp = obbox.get_point_indices_within_bounding_box(points)
            overlap_point_ids = overlap_point_ids.union(set(tmp))
        overlap_ratio = len(overlap_point_ids) / num_points

        if return_indices:
            return overlap_ratio, overlap_point_ids
        
        return overlap_ratio