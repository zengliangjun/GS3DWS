from torch import nn
from utils import loss_utils
import torch

class Loss(nn.Module):
    def __init__(self, _config):        
        super(Loss, self).__init__()
        self.config = _config
        self.optimConfg = _config.optim

    def forward(self, _input, _renderout):
        _color = _renderout['color']
        _gcolor = _input.colorTensor

        _l1loss = loss_utils.l1_loss(_color, _gcolor)
        _ssimloss = 1.0 - loss_utils.ssim(_color, _gcolor)
        _loss = (1.0 - self.optimConfg.lambda_dssim) * _l1loss + self.optimConfg.lambda_dssim * _ssimloss

        _iter = 0
        if hasattr(self.config, "iter"):
            _iter = self.config.iter

        # regularization
        _lambda_normal = self.optimConfg.lambda_normal if _iter > 7000 else 0.0
        _lambda_dist = self.optimConfg.lambda_dist if _iter > 3000 else 0.0

        _rdist = _renderout["rdist"]
        _rnormal  = _renderout['rnormal']
        _snormal = _renderout['snormal']

        _normal_error = (1 - torch.sum(_rnormal * _snormal, dim = 0)).mean()
        _rdist = _rdist.mean()

        if 0 != _lambda_normal:
            _loss += _lambda_normal * _normal_error
        
        if 0 != _lambda_dist:
            _loss += _lambda_dist * _rdist

        return _loss, {'l1': _l1loss, \
                       "ssim": _ssimloss, \
                       "normal": _normal_error, \
                       "dist": _rdist}
