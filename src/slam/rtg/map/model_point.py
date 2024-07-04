from gausiansplatting.scene import gs_model
from common.raw import points as rawpoints
from utils import sh_utils, geometry_utils

import torch

class MapState():

    def __init__(self, args) -> None:
        super(MapState, self).__init__()
        # self.normals = None
        self.confidences = None
        self.add_ticks = None
        self.depth_error_counters = None
        self.color_error_counters = None

        self.field_keys = ["confidences", "add_ticks", "depth_error_counters", "color_error_counters"]

    def update(self, _mask, _params = None):
        if self.confidences is None:
            return

        if _params is not None:
            for _key in self.field_keys:
                _params[_key] = getattr(self, _key)[_mask]

        _mask = torch.logical_not(_mask)
        for _key in self.field_keys:
            _values = getattr(self, _key)[_mask]
            setattr(self, _key, _values)

    def merger(self, _paramters):
        if self.confidences is None:
            for _key in self.field_keys:
                _values = _paramters[_key]
                setattr(self, _key, _values)
        else:
            for _key in self.field_keys:
                _svalues = getattr(self, _key)
                _pvalues = _paramters[_key]
                setattr(self, _key, torch.cat((_svalues, _pvalues), dim = 0))

    def updateCount(self, _mask):
        self.confidences[_mask] += 1


class MapClouds(gs_model.Module):

    def __init__(self, _config):
        super(MapClouds, self).__init__(_config)
        self.state = MapState(_config)
        self.gaussianConfg = _config.mapper.gaussian

    def build(self, _points: rawpoints.RawPointsTensor):
        '''
        Can't update module
        '''
        _colors = sh_utils.RGB2SHTensor(_points.colors)
        _features = torch.zeros((_colors.shape[0], 3, (self.sceneConf.sh_degree + 1) ** 2),
                                dtype = _colors.dtype, device = _colors.device)
        _features[:, :3, 0 ] = _colors

        if (
            self.gaussianConfg.xyz_factor[0] == 1
            and self.gaussianConfg.xyz_factor[1] == 1
            and self.gaussianConfg.xyz_factor[2] == 1
        ):
            _rots = torch.zeros((_colors.shape[0], 4), dtype= _colors.dtype, device= _colors.device)
            _rots[:, 0] = 1
        else:
            _z_axis = torch.tensor([0, 0, 1], dtype= _colors.dtype, device= _colors.device).repeat(_colors.shape[0], 1)
            _rots = geometry_utils.compute_rot(_z_axis, _points.normals)


        _opacities = self.gaussianConfg.init_opacity * \
                    torch.ones((_colors.shape[0], 1),  \
                        dtype = _colors.dtype, device = _colors.device)

        _scaling = torch.ones((_colors.shape[0], 3), dtype = _colors.dtype, device= _colors.device) * 1e-6

        return {
                'xyz': _points.points,
                'features_dc': _features[ :, :, 0:1].transpose(1, 2).contiguous(),
                'features_rest': _features[ :, :, 1:].transpose(1, 2).contiguous(),
                'opacities': self.points.opacity_inverse_activation(_opacities),
                'scaling': self.points.scaling_inverse_activation(_scaling),
                'rotation': _rots,
            }

    def buildWithId(self, _points: rawpoints.RawPointsTensor, _timestamp):
        if not hasattr(_points, "normals"):
            return None

        ## TODO
        _mag = torch.norm(_points.normals, p=2, dim=-1, keepdim=True)
        _points.normals = _points.normals / (_mag + 1e-8)

        _valid_mask = _points.normals.sum(dim=-1) != 0
        _points.points = _points.points[_valid_mask]
        _points.colors = _points.colors[_valid_mask]
        _points.normals = _points.normals[_valid_mask]

        _params = self.build(_points)

        _points_num = _valid_mask.sum()
        _dtype = _points.points.dtype
        _device = _points.points.device

        _confidences = torch.zeros([_points_num, 1], dtype = _dtype, device = _device)
        _add_ticks = _timestamp * torch.ones([_points_num, 1], dtype = _dtype, device = _device)

        _depth_error_counters = torch.zeros([_points_num, 1], dtype = _dtype, device = _device)
        _color_error_counters = torch.zeros([_points_num, 1], dtype = _dtype, device = _device)

        #_params["normals"] = _points.normals
        _params["confidences"] = _confidences
        _params["add_ticks"] = _add_ticks
        _params["depth_error_counters"] = _depth_error_counters
        _params["color_error_counters"] = _color_error_counters

        return _params

    def counter_update(self, _color_mask, _depth_mask):
        self.state.color_error_counters[_color_mask] += 1
        self.state.depth_error_counters[_depth_mask] += 1

    def counter_calcute(self, delete_thresh = 10):        
        color_mask = (self.state.color_error_counters >= delete_thresh).squeeze()
        depth_mask = (self.state.depth_error_counters >= delete_thresh).squeeze()
        return color_mask, depth_mask

    def reset(self, _mask, _timestamp):
        if _mask.sum() <= 0:
            return
        
        self.state.confidences[_mask] = 0
        self.state.add_ticks[_mask] = _timestamp

    def updateCount(self):
        _grad_mask = (self.points.features_dc.grad.abs() != 0).any(dim=-1)
        self.state.updateCount(_grad_mask)

    def renderparams(self):
        if self.xyz is None:
            return None

        _params = super(MapClouds, self).renderparams()
        _params["radius"] = self.radius
        _params["normals"] = self.normals
        return _params

    @property
    def radius(self):
        if self.xyz is None:
            return None
        scales = self.scaling
        min_length, _ = torch.min(scales, dim=1)
        radius = (torch.sum(scales, dim=1) - min_length) / 2
        return radius

    @property
    def R(self):
        if self.xyz is None:
            return None
        from pytorch3d.transforms import quaternion_to_matrix
        return quaternion_to_matrix(self.rotation)

    @property
    def normals(self):
        if self.xyz is None:
            return None
        scales = self.scaling
        R = self.R
        min_indices = torch.argmin(scales, dim=1)
        normal = torch.gather(
            R.transpose(1, 2),
            1,
            min_indices.unsqueeze(1).unsqueeze(2).expand(-1, -1, 3),
        )
        normal = normal[:, 0, :]
        mag = torch.norm(normal, p=2, dim=-1, keepdim=True)
        return normal / (mag + 1e-8)
