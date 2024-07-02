from gausiansplatting.scene import gs_policy
from common.sys import gsmodel, gsoptimer
import torch

class Policy(gs_policy.Policy):
    def __init__(self, _config) -> None:
        super(Policy, self).__init__(_config)
        self.optimConfg = _config.optim

    def _prune(self, _iter, _model: gsmodel.IModule):        
        _prune_mask = super(Policy, self)._prune(_iter, _model)
        _mask = torch.sigmoid(_model.points.mask).squeeze() <= 0.01
        return torch.logical_or(_prune_mask, _mask)

    def _pruneWithMask(self, _model: gsmodel.IModule, _optimer: gsoptimer.IOptimer):
        _mask = (torch.sigmoid(_model.points.mask).squeeze() <= 0.01)
        _model.prune(_mask)
        _params = _model.params()
        _optimer.update(_params)

    def _prunePreUpdate(self, _iter: int, _model: gsmodel.IModule, _optimer: gsoptimer.IOptimer):
        if _iter != self.optimConfg.iterations:
            return
        self._pruneWithMask(_model, _optimer)

    def _prunePostUpdate(self, _iter: int, _model: gsmodel.IModule, _optimer: gsoptimer.IOptimer):
        if _iter < self.policyConfg.densify_until_iter or \
            _iter % self.policyConfg.mask_prune_iter != 0:
            return
        self._pruneWithMask(_model, _optimer)

    def update(self, _iter: int, _model: gsmodel.IModule, _optimer: gsoptimer.IOptimer, _rout: dict):
        self._prunePreUpdate(_iter, _model, _optimer)
        super(Policy, self).update(_iter, _model, _optimer, _rout)
        self._prunePostUpdate(_iter, _model, _optimer)
