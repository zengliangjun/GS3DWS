from gausiansplatting.scene import gs_policy
from common.sys import gsmodel, gsoptimer

class Policy(gs_policy.Policy):

    def __init__(self, _config) -> None:
        super(Policy, self).__init__(_config)

    def update(self, _iter: int, _model: gsmodel.IModule, _optimer: gsoptimer.IOptimer, _rout: dict):
        setattr(self.config, "iter", _iter)
        super(Policy, self).update(_iter, _model, _optimer, _rout)

