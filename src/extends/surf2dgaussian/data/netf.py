from gausiansplatting.data import nerf
import os.path as osp
import cv2
from utils import camera_utils

class NerfSource(nerf.NerfSource):

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
        _frame.color = _color