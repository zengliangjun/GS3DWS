from common.gs import points_utils
from simple import logger
import numpy as np
from plyfile import PlyData, PlyElement
import torch
from torch import nn

class GSPoints(points_utils.ModuleBase):

    def __init__(self, args) -> None:
        super(GSPoints, self).__init__()

        self.config = args

        self.xyz = None
        self.features_dc = None
        self.features_rest = None
        self.scaling = None
        self.rotation = None
        self.opacities = None

    def update(self, _mask, _params = None):
        if self.xyz is None:
            return

        if _params is not None:
            _params["xyz"] = self.xyz[_mask]
            _params["features_dc"] = self.features_dc[_mask]
            _params["features_rest"] = self.features_rest[_mask]
            _params["scaling"] = self.scaling[_mask]
            _params["rotation"] = self.rotation[_mask]
            _params["opacities"] = self.opacities[_mask]

        _mask = torch.logical_not(_mask)
        self.xyz = self.xyz[_mask]
        self.features_dc = self.features_dc[_mask]
        self.features_rest = self.features_rest[_mask]
        self.scaling = self.scaling[_mask]
        self.rotation = self.rotation[_mask]
        self.opacities = self.opacities[_mask]

    def merger(self, _paramters):
        if "xyz" not in _paramters:
            return

        _xyz = _paramters['xyz']
        _features_dc = _paramters['features_dc']
        _features_rest = _paramters['features_rest']
        _scaling = _paramters['scaling']
        _rotation = _paramters['rotation']
        _opacities = _paramters['opacities']

        if self.xyz is None:
            self.xyz = _xyz
            self.features_dc = _features_dc
            self.features_rest = _features_rest
            self.scaling = _scaling
            self.rotation = _rotation
            self.opacities = _opacities
        else:
            _requires_grad = self.xyz.requires_grad

            _isParamer = isinstance(self.xyz, nn.Parameter)

            self.xyz = torch.cat((self.xyz, _xyz), dim = 0)
            self.features_dc = torch.cat((self.features_dc, _features_dc), dim = 0)
            self.features_rest = torch.cat((self.features_rest, _features_rest), dim = 0)
            self.scaling = torch.cat((self.scaling, _scaling), dim = 0)
            self.rotation = torch.cat((self.rotation, _rotation), dim = 0)
            self.opacities = torch.cat((self.opacities, _opacities), dim = 0)

            if _isParamer:
                self.xyz = nn.Parameter(self.xyz)
                self.opacities = nn.Parameter(self.opacities)
                self.features_dc = nn.Parameter(self.features_dc)
                self.features_rest = nn.Parameter(self.features_rest)
                self.scaling = nn.Parameter(self.scaling)
                self.rotation = nn.Parameter(self.rotation)

            self.train(_grad = _requires_grad)

    def set(self, _key, _value):
        assert self.xyz is not None
        assert self.xyz.shape[0] == _value.shape[0]
        assert hasattr(self, _key)
        _orgValue = getattr(self, _key)
        _isParamer = isinstance(_orgValue, nn.Parameter)

        _value.requires_grad = _orgValue.requires_grad
        if _isParamer:
            _value = nn.Parameter(_value)

        setattr(self, _key, _value)
        del _orgValue

    def clone(self, _mask):
        return {
            'xyz': self.xyz[_mask],
            'opacities': self.opacities[_mask],
            'features_dc': self.features_dc[_mask],
            'features_rest': self.features_rest[_mask],
            'scaling': self.scaling[_mask],
            'rotation': self.rotation[_mask]
        }

    def train(self, _grad = True):
        self.xyz.requires_grad = _grad
        self.features_dc.requires_grad = _grad
        self.features_rest.requires_grad = _grad
        self.scaling.requires_grad = _grad
        self.rotation.requires_grad = _grad
        self.opacities.requires_grad = _grad

    def eval(self):
        self.train(_grad = False)

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self.features_dc.shape[1] * self.features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self.features_rest.shape[1] * self.features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self.scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self.rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def params(self):
        if not isinstance(self.xyz, nn.Parameter):
            self.xyz = nn.Parameter(self.xyz)
            self.opacities = nn.Parameter(self.opacities)
            self.features_dc = nn.Parameter(self.features_dc)
            self.features_rest = nn.Parameter(self.features_rest)
            self.scaling = nn.Parameter(self.scaling)
            self.rotation = nn.Parameter(self.rotation)

        return {
            'xyz': self.xyz,
            'opacities': self.opacities,
            'features_dc': self.features_dc,
            'features_rest': self.features_rest,
            'scaling': self.scaling,
            'rotation': self.rotation
        }

    def save(self, _path: str):
        if self.xyz is None:
            logger.log(f'can\'t save {_path} with points is None')
            return

        _xyz = self.xyz.detach().cpu().numpy()
        _f_dc = self.features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        _f_rest = self.features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

        _opacities = self.opacities.detach().cpu().numpy()
        _scaling = self.scaling.detach().cpu().numpy()
        _rotation = self.rotation.detach().cpu().numpy()

        _normals = np.zeros_like(_xyz)

        _dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        _elements = np.empty(_xyz.shape[0], dtype = _dtype_full)
        _attributes = np.concatenate((_xyz, _normals, _f_dc, _f_rest, _opacities, _scaling, _rotation), axis = 1)
        _elements[:] = list(map(tuple, _attributes))
        _el = PlyElement.describe(_elements, 'vertex')
        PlyData([_el]).write(_path)

    @staticmethod
    def load(_path: str, _sh_degree: int):
        _plydata = PlyData.read(_path)
        _xyz = np.stack((np.asarray(_plydata.elements[0]["x"]),
                         np.asarray(_plydata.elements[0]["y"]),
                         np.asarray(_plydata.elements[0]["z"])),  axis=1)
        _opacities = np.asarray(_plydata.elements[0]["opacity"])[..., np.newaxis]

        _f_dc = np.zeros((_xyz.shape[0], 3, 1))
        _f_dc[:, 0, 0] = np.asarray(_plydata.elements[0]["f_dc_0"])
        _f_dc[:, 1, 0] = np.asarray(_plydata.elements[0]["f_dc_1"])
        _f_dc[:, 2, 0] = np.asarray(_plydata.elements[0]["f_dc_2"])
        _f_dc = _f_dc.transpose(0, 2, 1)

        _restnames = [p.name for p in _plydata.elements[0].properties if p.name.startswith("f_rest_")]
        _restnames = sorted(_restnames, key = lambda x: int(x.split('_')[-1]))
        assert len(_restnames) == 3 * (_sh_degree + 1) ** 2 - 3
        _f_rest = np.zeros((_xyz.shape[0], len(_restnames)))
        for _idx, _attr_name in enumerate(_restnames):
            _f_rest[:, _idx] = np.asarray(_plydata.elements[0][_attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        _f_rest = _f_rest.reshape((_f_rest.shape[0], 3, (_sh_degree + 1) ** 2 - 1))
        _f_rest = _f_rest.transpose(0, 2, 1)

        _scale_names = [p.name for p in _plydata.elements[0].properties if p.name.startswith("scale_")]
        _scale_names = sorted(_scale_names, key = lambda x: int(x.split('_')[-1]))
        _scaling = np.zeros((_xyz.shape[0], len(_scale_names)))
        for idx, attr_name in enumerate(_scale_names):
            _scaling[:, idx] = np.asarray(_plydata.elements[0][attr_name])

        _rot_names = [p.name for p in _plydata.elements[0].properties if p.name.startswith("rot")]
        _rot_names = sorted(_rot_names, key = lambda x: int(x.split('_')[-1]))
        _rots = np.zeros((_xyz.shape[0], len(_rot_names)))
        for idx, attr_name in enumerate(_rot_names):
            _rots[:, idx] = np.asarray(_plydata.elements[0][attr_name])

        return {
            'xyz': torch.tensor(_xyz, dtype= torch.float32),
            'opacities': torch.tensor(_opacities, dtype= torch.float32),
            'features_dc': torch.tensor(_f_dc, dtype= torch.float32),
            'features_rest': torch.tensor(_f_rest, dtype= torch.float32),
            'scaling': torch.tensor(_scaling, dtype= torch.float32),
            'rotation': torch.tensor(_rots, dtype= torch.float32),
        }
