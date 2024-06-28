import numpy as np
import torch

from plyfile import PlyData
import numpy as np
from utils import sh_utils
import os.path as osp

def _fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T

    _rpoints = RawPoints()
    _rpoints.points = positions
    _rpoints.colors = colors
    _rpoints.normals = normals

    return _rpoints

class RawPoints():
    points : np.array
    colors : np.array
    normals : np.array

    def tensor(self):
        _rpoints = RawPointsTensor()
        _rpoints.points = torch.tensor(self.points, dtype= torch.float32)
        _rpoints.colors = torch.tensor(self.colors, dtype= torch.float32)
        _rpoints.normals = torch.tensor(self.normals, dtype= torch.float32)
        return _rpoints

    @staticmethod
    def load(_path):
        try:
            _pcd = _fetchPly(_path)
        except:
            _pcd = None

        if _pcd is None:
            # Since this data set has no colmap data, we start with random points
            num_pts = 100_000
            # We create random points inside the bounds of the synthetic Blender scenes
            xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
            shs = np.random.random((num_pts, 3)) / 255.0
            _pcd = RawPoints()
            _pcd.points = xyz
            _pcd.colors = sh_utils.SH2RGB(shs)
            _pcd.normals = np.zeros((num_pts, 3))
        return _pcd

class RawPointsTensor():
    points : torch.tensor
    colors : torch.tensor
    normals : torch.tensor

    def numpy(self):
        _rpoints = RawPoints()
        _rpoints.points = self.points.numpy()
        _rpoints.colors = self.colors.numpy()
        _rpoints.normals = self.normals.numpy()
        return _rpoints

    def to(self, _device):
        self.points = self.points.to(_device)
        self.colors = self.colors.to(_device)
        self.normals = self.normals.to(_device)
        return self

