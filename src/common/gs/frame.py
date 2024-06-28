from common.raw import frame
import numpy as np
import torch
import math

class GSFrame():

    def __init__(self, _frame: frame.Frame) -> None:
        self.data = _frame

    def to(self, _device):
        if not hasattr(self, "colorTensor"):
            self.colorTensor = torch.tensor(self.color / 255, dtype = torch.float32).permute(2, 0, 1) # c*h*w

        self.colorTensor = self.colorTensor.to(_device) 
        return self

    @property
    def id(self):
        return self.data.id

    @property
    def fovY(self):
        return self.data.fovY

    @property
    def fovX(self):
        return self.data.fovX

    @property
    def cx(self):
        return self.data.cx

    @property
    def cy(self):
        return self.data.cy

    @property
    def width(self):
        return self.data.width

    @property
    def height(self):
        return self.data.height

    @property
    def w2cRGT(self):
        return self.data.w2cR

    @property
    def w2cTGT(self):
        return self.data.w2cT

    @property
    def w2cGT(self):
        w2c = np.zeros((4, 4), dtype= np.float32)
        w2c[:3, :3] = self.w2cRGT
        w2c[:3, 3] = self.w2cTGT
        w2c[3, 3] = 1.0
        return w2c

    @property
    def w2cTensor(self):
        if hasattr(self, "w2c") and self.w2c is not None:
            return torch.tensor(self.w2c)
        else:
            return torch.tensor(self.w2cGT)

    @property
    def c2w(self):
        if hasattr(self, "w2c") and self.w2c is not None:
            return np.linalg.inv(self.w2c).astype(np.float32)
        else:
            return None

    @property
    def c2wGT(self):
        return np.linalg.inv(self.w2cGT).astype(np.float32)

    @property
    def c2wTensor(self):
        if hasattr(self, "c2w") and self.c2w is not None:
            return torch.tensor(self.c2w)
        else:
            return torch.tensor(self.c2wGT, dtype= torch.float32)

    @property
    def c2wCenterTensor(self):
        return self.c2wTensor[:3, 3]

    @property
    def projectView(self):
        _zfar = 100.0
        _znear = 0.01

        tanHalfFovY = math.tan((self.fovY / 2))
        tanHalfFovX = math.tan((self.fovX / 2))

        top = tanHalfFovY * _znear
        bottom = -top
        right = tanHalfFovX * _znear
        left = -right

        P = np.zeros((4, 4), dtype= np.float32)

        z_sign = 1.0

        P[0, 0] = 2.0 * _znear / (right - left)
        P[1, 1] = 2.0 * _znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * _zfar / (_zfar - _znear)
        P[2, 3] = -(_zfar * _znear) / (_zfar - _znear)
        return P

    @property
    def projectViewTensor(self):
        return torch.tensor(self.projectView)

    @property
    def w2cViewTensor(self):
        #_result = torch.mm(self.w2cTensor.T, self.projectViewTensor.T).T
        _result = torch.mm(self.projectViewTensor, self.w2cTensor)
        return _result

    @property
    def color(self):
        return self.data.color

    @property
    def color_name(self):
        return self.data.color_name

    @property
    def pixel_num(self):
        return self.width * self.height
