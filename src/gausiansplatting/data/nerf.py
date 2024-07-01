import os.path as osp
import numpy as np
import json
from common.raw import points, frame
import random
import cv2
from utils import camera_utils

class NerfSource:

    def _loadfull(self, _frame):
        _path = osp.join(self.dataConf.source_path, _frame.color_name + ".png")

        _color = cv2.imread(_path, cv2.IMREAD_UNCHANGED)
        _color = _color[:, :, [2, 1, 0, 3]]

        fovy = camera_utils.focal2fov(camera_utils.fov2focal(self.camera_angle_x, _color.shape[1]), _color.shape[0])

        _frame.fovY = fovy
        _frame.fovX = self.camera_angle_x
        _frame.cx = _color.shape[1] / 2  ###
        _frame.cy = _color.shape[0] / 2  ###
        _frame.width = _color.shape[1]
        _frame.height = _color.shape[0]
        _frame.color = _color[:, :, :3]

    def _loadframes(self, _path):
        _frameinfos = []

        with open(_path, 'r') as _fd:
            contents = json.load(_fd)
            self.camera_angle_x = contents["camera_angle_x"]
            _frames = contents["frames"]
            for _idx, _frame in enumerate(_frames):
                _name = _frame["file_path"]

                # NeRF 'transform_matrix' is a camera-to-world transform
                _c2w = np.array(_frame["transform_matrix"])
                # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
                _c2w[:3, 1:3] *= -1

                # get the world-to-camera transform and set R, T
                _w2c = np.linalg.inv(_c2w)
                _R = _w2c[:3, :3] #np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
                _T = _w2c[:3, 3]

                _frame = frame.Frame()
                _frame.id = _idx
                _frame.fovY = 0
                _frame.fovX = 0
                _frame.cx = 0
                _frame.cy = 0
                _frame.width = 0
                _frame.height = 0

                _frame.w2cR = _R
                _frame.w2cT = _T

                _frame.color_name = _name
                _frame.color = None

                _frameinfos.append(_frame)

            self.frames = _frameinfos

    def __init__(self, _config):
        self.dataConf = _config.scene.data

        if self.dataConf.eval:
            _camefile = "transforms_test.json"
        else:
            _camefile = "transforms_train.json"

        if not osp.isabs(self.dataConf.source_path):
            _root = osp.dirname(osp.abspath(__file__))
            print(_root)
            _root = osp.abspath(osp.join(_root, "../../../"))
            print(_root)
            self.dataConf.source_path = osp.join(_root, self.dataConf.source_path)

        _path = osp.join(self.dataConf.source_path, _camefile)

        self._loadframes(_path)
        self.idxes = [_i for _i in range(len(self.frames))]

        if not self.dataConf.eval:
            random.shuffle(self.idxes)
            random.shuffle(self.idxes)

        self.currentids = 0
        self.nerf_normalization = self._nerfppNorm()


    def __len__(self):
        return len(self.idxes)

    def __iter__(self):
        return self

    def __next__(self):
        if self.currentids >= len(self):
            if self.dataConf.eval:
                return None

            random.shuffle(self.idxes)
            random.shuffle(self.idxes)
            self.currentids = 0

        _currentids = self.currentids
        self.currentids += 1

        _idx = self.idxes[_currentids]

        _frame = self.frames[_idx]
        if _frame.color is None:
            self._loadfull(_frame)
        return _frame

    def _nerfppNorm(self):
        def get_center_and_diag(cam_centers):
            cam_centers = np.hstack(cam_centers)
            avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
            center = avg_cam_center
            dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
            diagonal = np.max(dist)
            return center.flatten(), diagonal

        _centers = []

        for _frame in self.frames:
            _c2w = _frame.c2w
            _centers.append(_c2w[:3, 3:4])
        _center, _diagonal = get_center_and_diag(_centers)
        _radius = _diagonal * 1.1

        _translate = - _center
        return {"translate": _translate, "radius": _radius}

    def loadRawPoints(self):
        _ply_path = osp.join(self.dataConf.source_path, "points3d.ply")
        return points.RawPoints.load(_ply_path)

