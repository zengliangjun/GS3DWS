import numpy as np
from utils import camera_utils

class Frame():

    id: int
    fovY: float
    fovX: float
    cx: int
    cy: int
    width: int
    height: int

    w2cR: np.array # 3x3  GT
    w2cT: np.array # 3    GT

    color_name: str
    color: np.array  # h*w*c

    @property
    def intrinsic(self):
        w, h = self.width, self.height
        fx, fy = camera_utils.fov2focal(self.foVx, w), camera_utils.fov2focal(self.foVy, h)
        cx = self.cx if self.cx > 0 else w / 2
        cy = self.cy if self.cy > 0 else h / 2
        _intrinstic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype= np.float32)
        return _intrinstic

    @property
    def w2c(self):
        w2cRt = np.zeros((4, 4))
        w2cRt[:3, :3] = self.w2cR
        w2cRt[:3, 3] = self.w2cT
        w2cRt[3, 3] = 1.0
        return w2cRt

    @property
    def c2w(self):
        return np.linalg.inv(self.w2c)

