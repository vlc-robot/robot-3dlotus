import os
from typing import Type
import numpy as np

from pathlib import Path
from typing import Tuple, Dict, List
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor


class CameraMotion(object):
    def __init__(self, cam: VisionSensor):
        self.cam = cam

    def step(self):
        raise NotImplementedError()

    def save_pose(self):
        self._prev_pose = self.cam.get_pose()

    def restore_pose(self):
        self.cam.set_pose(self._prev_pose)


class CircleCameraMotion(CameraMotion):

    def __init__(self, cam: VisionSensor, origin: Dummy, speed: float):
        super().__init__(cam)
        self.origin = origin
        self.speed = speed  # in radians

    def step(self):
        self.origin.rotate([0, 0, self.speed])


class StaticCameraMotion(CameraMotion):

    def __init__(self, cam: VisionSensor):
        super().__init__(cam)

    def step(self):
        pass

class AttachedCameraMotion(CameraMotion):

    def __init__(self, cam: VisionSensor, parent_cam: VisionSensor):
        super().__init__(cam)
        self.parent_cam = parent_cam

    def step(self):
        self.cam.set_pose(self.parent_cam.get_pose())


class TaskRecorder(object):

    def __init__(self, cams_motion: Dict[str, CameraMotion], fps=30):
        self._cams_motion = cams_motion
        self._fps = fps
        self._snaps = {cam_name: [] for cam_name in self._cams_motion.keys()}

    def take_snap(self):
        for cam_name, cam_motion in self._cams_motion.items():
            cam_motion.step()
            self._snaps[cam_name].append(
                (cam_motion.cam.capture_rgb() * 255.).astype(np.uint8))

    def save(self, path):
        print('Converting to video ...')
        path = Path(path)
        path.mkdir(exist_ok=True)
        # OpenCV QT version can conflict with PyRep, so import here
        import cv2
        for cam_name, cam_motion in self._cams_motion.items():
            video = cv2.VideoWriter(
                    str(path / f"{cam_name}.avi"), cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), self._fps,
                    tuple(cam_motion.cam.get_resolution()))
            for image in self._snaps[cam_name]:
                video.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            video.release()

        self._snaps = {cam_name: [] for cam_name in self._cams_motion.keys()}
