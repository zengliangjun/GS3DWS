from typing import Any
import os.path as osp
import yaml
import numpy as np
from utils import camera_utils
import cv2
from common.raw import frame

class RGBDSource:

    def _associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """pair images, depths, and poses"""
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if np.abs(tstamp_depth[j] - t) < max_dt:
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and (
                    np.abs(tstamp_pose[k] - t) < max_dt
                ):
                    associations.append((i, j, k))

        return associations

    def _filter_frames(self, tstamp_image, associations, frame_num = -1, frame_start=0, frame_step=0):
        indicies = [0]
        frame_rate = 32
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        n_img = len(indicies)
        if frame_num == -1:
            indexs = list(range(n_img))
        else:
            indexs = list(range(frame_num))
        indicies = [frame_start + i * (frame_step + 1) for i in indexs]
        indicies = [i for i in indicies if i < n_img]
        return indicies

    def __init__(self, _config) -> None:
        _config = _config.scene.data

        """ read video data in tum-rgbd format """
        if osp.isfile(osp.join(_config.source_path, "groundtruth.txt")):
            _pose_list = osp.join(_config.source_path, "groundtruth.txt")
        elif osp.isfile(osp.join(_config.source_path, "pose.txt")):
            _pose_list = osp.join(_config.source_path, "pose.txt")

        _config_path = osp.join(_config.source_path, "config.yaml")
        with open(_config_path, "r") as f:
            _dataconfig = yaml.load(f, Loader=yaml.FullLoader)

        _image_list = osp.join(_config.source_path, "rgb.txt")
        _depth_list = osp.join(_config.source_path, "depth.txt")

        _image_data = np.loadtxt(_image_list, delimiter=" ", dtype=np.unicode_)
        _depth_data = np.loadtxt(_depth_list, delimiter=" ", dtype=np.unicode_)
        _pose_data = np.loadtxt(_pose_list, delimiter=" ", dtype=np.unicode_, skiprows=1)
        _pose_vecs = _pose_data[:, 1:].astype(np.float64)

        _tstamp_image = _image_data[:, 0].astype(np.float64)
        _tstamp_depth = _depth_data[:, 0].astype(np.float64)
        _tstamp_pose = _pose_data[:, 0].astype(np.float64)
        _associations = self._associate_frames(_tstamp_image, _tstamp_depth, _tstamp_pose)
        _indicies = self._filter_frames(_tstamp_image, _associations)

        color_paths, depth_paths, poses, timestamps = [], [], [], []
        inv_pose = None
        rgbd_pose_tupe = []
        for idx in range(len(_indicies)):

            ix = _indicies[idx]
            (i, j, k) = _associations[ix]
            color_paths += [_image_data[i, 1]]
            depth_paths += [_depth_data[j, 1]]
            rgbd_pose_tupe.append([_image_data[i, 1], _depth_data[j, 1], _tstamp_pose[k]])
            c2w = camera_utils.posev3quatv4_tomatrix(_pose_vecs[k])
            if inv_pose is None:
                inv_pose = np.linalg.inv(c2w)
                c2w = np.eye(4, dtype= np.float64)
            else:
                c2w = inv_pose @ c2w
            poses += [c2w]
            timestamps.append(_tstamp_image[i])

        # read config
        self.depth_scale = _dataconfig["depth_scale"]
        self.intrinsic = np.array(
            [[_dataconfig["fx"], 0, _dataconfig["cx"]],
             [0, _dataconfig["fy"], _dataconfig["cy"]],
             [0, 0, 1]]
        )
        self.crop_edge = _dataconfig["crop_edge"]

        self.color_paths = color_paths
        self.depth_paths = depth_paths
        self.poses = poses
        self.timestamps = timestamps

        self.config = _config
        self.currentids = 0


    def __len__(self):
        return len(self.color_paths)

    def __iter__(self):
        self.currentids = 0
        return self

    def __next__(self):
        if self.currentids >= len(self):
            return None

        _currentids = self.currentids
        self.currentids += 1

        c2w = self.poses[_currentids]
        if _currentids == 0:
            self.pose_w_t0 = np.linalg.inv(c2w)
        # pass invalid pose
        if np.isinf(c2w).any():
            # TODO
            return next(self)

        if not self.config.eval:
            c2w = self.pose_w_t0 @ c2w
        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3] #np.transpose()

        # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        _color = cv2.imread(osp.join(self.config.source_path, self.color_paths[_currentids]), cv2.IMREAD_UNCHANGED)
        _color = _color[:, :, :: -1]
        _depth = cv2.imread(osp.join(self.config.source_path, self.depth_paths[_currentids]), cv2.IMREAD_UNCHANGED) / self.depth_scale
        _color = cv2.resize(_color, (_depth.shape[1], _depth.shape[0]))

        _fx, _fy = self.intrinsic[0, 0], self.intrinsic[1, 1]
        _cx, _cy = self.intrinsic[0, 2], self.intrinsic[1, 2]
        if self.crop_edge > 0:
            _color = _color[
                self.crop_edge:-self.crop_edge,
                self.crop_edge:-self.crop_edge,
                :,
            ]
            _depth = _depth[
                self.crop_edge:-self.crop_edge,
                self.crop_edge:-self.crop_edge,
            ]
            _cx -= self.crop_edge
            _cy -= self.crop_edge

        height, width = _color.shape[:2]
        # print("image size:", height, width)
        FovX = camera_utils.focal2fov(_fx, width)
        FovY = camera_utils.focal2fov(_fy, height)
        _color_name = self.color_paths[_currentids]
        _depth_name = self.depth_paths[_currentids]
        #_timestamp = self.timestamps[_currentids]

        _data = frame.FrameRGBD()

        _data.id = _currentids
        _data.fovY = FovY
        _data.fovX = FovX
        _data.cx = _cx
        _data.cy = _cy
        _data.width = width
        _data.height = height

        _data.w2cR = R.astype(np.float32)
        _data.w2cT = T.astype(np.float32)

        _data.color = _color
        _data.color_name = _color_name

        _data.depth = _depth
        _data.depth_name = _depth_name
        _data.depth_scale = self.depth_scale

        return _data

import hydra
_rootdir = osp.join(osp.dirname(__file__), "../../../../")
_rootdir = osp.abspath(_rootdir)

from omegaconf import DictConfig
from gausiansplatting import config

def load_data_config(cfg: DictConfig) -> config.DataCfg:
    return config.load_typed_config(cfg, config.DataCfg, {})


@hydra.main(
    version_base=None,
    config_path=osp.join(_rootdir, "config/scene/data/"),
    config_name="tum_freiburg3"
)
def test(cfg_dict: DictConfig):
    _config = load_data_config(cfg_dict)

    _source = RGBDSource(_config)
    _iter = iter(_source)
    while True:
        _data = next(_iter)
        print(_data)


if __name__ == "__main__":
    test()
