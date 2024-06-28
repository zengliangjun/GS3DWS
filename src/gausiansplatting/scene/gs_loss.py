from torch import nn
from utils import loss_utils

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
        return _loss, {'l1': _l1loss, "ssim": _ssimloss}
