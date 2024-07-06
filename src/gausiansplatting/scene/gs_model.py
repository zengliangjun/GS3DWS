from common.sys import gsmodel
from common.gs import points as gspoints
from common.raw import points as rawpoints
from utils import sh_utils, general_utils
import torch
import simple_knn._C as sknn

class CountState():

    def __init__(self, args) -> None:
        super(CountState, self).__init__()
        self.confidences = None
        self.max_2dradii = None
        self.xyz_gradient_accum = None

        self.field_keys = ["confidences", "max_2dradii", "xyz_gradient_accum"]

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
        if "confidences" not in _paramters:
            _xyz = _paramters['xyz']

            _paramters["confidences"] = torch.zeros((_xyz.shape[0], 1), dtype = _xyz.dtype, device = _xyz.device)
            _paramters["max_2dradii"] = torch.zeros(_xyz.shape[0], dtype = _xyz.dtype, device = _xyz.device)
            _paramters["xyz_gradient_accum"] = torch.zeros((_xyz.shape[0], 1), dtype = _xyz.dtype, device = _xyz.device)

        if self.confidences is None:
            for _key in self.field_keys:
                _values = _paramters[_key]
                setattr(self, _key, _values)

        else:
            for _key in self.field_keys:
                _orgvalues = getattr(self, _key)
                _newvalues = _paramters[_key]
                _values = torch.cat((_orgvalues, _newvalues), dim = 0)
                setattr(self, _key, _values)

        for _key in self.field_keys:
            getattr(self, _key)[...] = 0

    def updateCount(self, _rout : dict):
        _mask = _rout['mask']
        self.max_2dradii[_mask] = torch.max(self.max_2dradii[_mask], _rout['2dradii'][_mask])
        self.xyz_gradient_accum[_mask, :] += torch.norm(_rout['uv'].grad[_mask, :2], dim=-1, keepdim=True)
        self.confidences[_mask, :] += 1

    def clear(self):
        for _key in self.field_keys:
            setattr(self, _key, None)

class Module(gsmodel.IModule):

    def __init__(self, _config):
        super().__init__()
        self.config = _config
        self.sceneConf = _config.scene

        #self.spatial_lr_scale = 1 ###
        self.points = gspoints.GSPoints(_config)
        self.state = CountState(_config)

        self.active_sh_degree = 0

    def oneupSHdegree(self):
        if self.active_sh_degree < self.sceneConf.sh_degree:
            self.active_sh_degree += 1


    def renderparams(self):
        if self.xyz is None:
            return None

        return {
            'xyz': self.xyz,
            'opacities': self.opacities,
            'features': self.features,
            'scaling': self.scaling,
            'rotation': self.rotation,
            'active_sh_degree': self.active_sh_degree
        }

    def params(self):
        return self.points.params()

    def train(self):
        self.points.train()

    def eval(self):
        self.points.eval()

    def prune(self, _mask):
        self.points.update(_mask)
        self.state.update(_mask)

    def split(self, _mask, _repeatN = 2):
        _params = self.clone(_mask)
        ###
        _org_scaling = _params['scaling'].repeat(_repeatN, 1)
        _scaling = self.points.scaling_activation(_org_scaling)

        _stds = _scaling
        _means = torch.zeros_like(_stds)
        _samples = torch.normal(mean = _means, std = _means)

        _rots = general_utils.build_rotation(_params['rotation']).repeat(_repeatN, 1, 1)

        _xyz = torch.bmm(_rots, _samples.unsqueeze(-1)).squeeze(-1) + _params['xyz'].repeat(_repeatN, 1)
        _scaling = _scaling / (0.8 * _repeatN)
        _rotation = _params['rotation'].repeat(_repeatN, 1)
        _features_dc = _params['features_dc'].repeat(_repeatN, 1, 1)
        _features_rest = _params['features_rest'].repeat(_repeatN, 1, 1)
        _opacities = _params['opacities'].repeat(_repeatN, 1)
        return {
            'xyz': _xyz,
            'opacities': _opacities,
            'features_dc': _features_dc,
            'features_rest': _features_rest,
            'scaling': self.points.scaling_inverse_activation(_scaling),
            'rotation': _rotation
        }

    def clone(self, _mask):
        return self.points.clone(_mask)

    def move(self, _mask):
        _params = {}
        self.points.update(_mask, _params)
        self.state.update(_mask, _params)
        return _params

    def merger(self, _paramters):
        if _paramters is None:
            return

        self.points.merger(_paramters)
        self.state.merger(_paramters)

    def reset(self, _keyAttr: str):
        if _keyAttr == "opacities":
            _opacities = self.opacities
            _opacities = torch.min(_opacities, torch.ones_like(_opacities) * 0.01)
            _opacities = self.points.opacity_inverse_activation(_opacities)
            self.points.set("opacities", _opacities)
        else:
            ### TODO
            assert False

    def build(self, _points: rawpoints.RawPointsTensor):
        '''
        Can't update module
        '''
        #self.spatial_lr_scale = spatial_lr_scale

        _colors = sh_utils.RGB2SHTensor(_points.colors)
        _features = torch.zeros((_colors.shape[0], 3, (self.sceneConf.sh_degree + 1) ** 2), dtype = _colors.dtype, device = _colors.device)
        _features[:, :3, 0 ] = _colors

        _dist2 = torch.clamp_min(sknn.distCUDA2(_points.points), 0.0000001)
        if hasattr(_points, "init_scaling"):
            _dist2 *= _points.init_scaling
        _scaling = torch.sqrt(_dist2)[...,None].repeat(1, 3)

        _rots = torch.zeros((_colors.shape[0], 4), dtype = _colors.dtype, device = _colors.device)
        _rots[:, 0] = 1

        if hasattr(_points, "init_opacities"):
            _init_opacities = _points.init_opacities
        else:
            _init_opacities = 0.1

        _opacities = torch.ones((_colors.shape[0], 1), dtype = _colors.dtype, device = _colors.device) * _init_opacities

        return {
                'xyz': _points.points,
                'opacities': self.points.opacity_inverse_activation(_opacities),
                'features_dc': _features[ :, :, 0:1].transpose(1, 2).contiguous(),
                'features_rest': _features[ :, :, 1:].transpose(1, 2).contiguous(),
                'scaling': self.points.scaling_inverse_activation(_scaling),
                'rotation': _rots,
            }

    def updateCount(self, _rout:dict):
        self.state.updateCount(_rout)

    def update(self, _paramters):
        for _key in _paramters:
            delattr(self.points, _key)
            setattr(self.points, _key, _paramters[_key])
        self.state.clear()
        self.state.merger(_paramters)

    def save(self, _path: str, _params: dict = None):
        self.points.save(_path)

    @property
    def points_num(self):
        if self.xyz is None:
            return 0
        return self.points.xyz.shape[0]

    @property
    def xyz(self):
        return self.points.xyz

    @property
    def scaling(self):
        if self.xyz is None:
            return None
        return self.points.scaling_activation(self.points.scaling)

    @property
    def rotation(self):
        if self.xyz is None:
            return None
        return self.points.rotation_activation(self.points.rotation)

    @property
    def opacities(self):
        return self.points.opacity_activation(self.points.opacities)

    @property
    def features(self):
        _dc = self.points.features_dc
        _rest = self.points.features_rest
        return torch.cat((_dc, _rest), dim = 1)
