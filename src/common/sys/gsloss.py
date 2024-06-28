from torch import nn


class ILoss(nn.Module):
    def __init__(self, _config):        
        super(ILoss, self).__init__()
        self.config = _config

    def forward(self, _input, _target):
        pass
