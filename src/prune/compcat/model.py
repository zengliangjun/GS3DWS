from gausiansplatting.scene import gs_model
from common.gs import points as gspoint
from common.raw import points as rawpoints

import torch
from torch import nn

class Points(gspoint.GSPoints):

    def __init__(self, args) -> None:
        super(Points, self).__init__(args)
        self.mask = None

    def update(self, _mask, _params = None):
        if self.mask is None:
            return
        super(Points, self).update(_mask, _params)

        if _params is not None:
            _params["mask"] = self.mask[_mask]

        _mask = torch.logical_not(_mask)
        self.mask = self.mask[_mask]

    def merger(self, _paramters):
        if "mask" not in _paramters:
            return

        _mask = _paramters['mask']

        if self.mask is None:
            self.mask = _mask
        else:
            _isParamer = isinstance(self.mask, nn.Parameter)
            self.mask = torch.cat((self.mask, _mask), dim = 0)
            if _isParamer:
                self.mask = nn.Parameter(self.mask)

        super(Points, self).merger(_paramters)

    def clone(self, _mask):
        _params = super(Points, self).clone(_mask)
        _params['mask'] = self.mask[_mask]
        return _params

    def train(self, _grad = True):
        super(Points, self).train(_grad)
        self.mask.requires_grad = _grad

    def params(self):
        _params = super(Points, self).params()
        if not isinstance(self.mask, nn.Parameter):
            self.mask = nn.Parameter(self.mask)
        _params['mask'] = self.mask
        return _params

class Model(gs_model.Module):
    def __init__(self, _config):
        super(Model, self).__init__(_config)
        self.points = Points(_config)

    def renderparams(self):
        _params = super(Model, self).renderparams()
        _params['mask'] = self.points.mask
        _mask = self.points.mask
        if _mask.requires_grad:
            _mask = torch.sigmoid(_params['mask'])
            _mask = ((_mask > 0.01).float() - _mask).detach() + _mask

            _params['opacities'] = _params['opacities'] * _mask
            _params['scaling'] = _params['scaling'] * _mask

        return _params

    def build(self, _points: rawpoints.RawPointsTensor):
        _paramers = super(Model, self).build(_points)
        _xyz = _points.points
        _paramers['mask'] = torch.ones((_xyz.shape[0], 1), dtype= _xyz.dtype, device = _xyz.device)
        return _paramers

    def split(self, _mask, _repeatN = 2):
        _paramers = super(Model, self).split(_mask, _repeatN)
        _paramers['mask'] = self.points.mask[_mask].repeat(_repeatN, 1)
        return _paramers
