from gausiansplatting.scene import gs_model
from common.raw import points as rawpoints
import torch
from utils import general_utils

class Module(gs_model.Module):

    def __init__(self, _config):
        super(Module, self).__init__(_config)


    def build(self, _points: rawpoints.RawPointsTensor):
        _parames = super(Module, self).build(_points)

        _parames['scaling'] = _parames['scaling'][:, :2]
        _orgRotation = _parames.pop("rotation")
        _parames['rotation'] = torch.rand_like(_orgRotation)
        return _parames

    def split(self, _mask, _repeatN = 2):
        _params = self.clone(_mask)
        ###
        _org_scaling = _params['scaling'].repeat(_repeatN, 1)
        _scaling = self.points.scaling_activation(_org_scaling)

        _stds = _scaling
        _stds = torch.cat([_stds, torch.zeros_like(_stds[:, :1])], dim=-1)
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