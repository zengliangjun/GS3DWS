from gausiansplatting.scene import gs_loss
import torch

class Loss(gs_loss.Loss):
    def __init__(self, _config):        
        super(Loss, self).__init__(_config)


    def forward(self, _input, _renderout):
        _loss, _items = super(Loss, self).forward(_input, _renderout)

        _paramters = getattr(_input, "paramters")
        _mloss = torch.mean((torch.sigmoid(_paramters['mask'])))
        _loss += _mloss * self.optimConfg.lambda_mask
        _items['mask'] = _mloss
        return _loss, _items
