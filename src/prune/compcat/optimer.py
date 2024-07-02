from gausiansplatting.scene import gs_optimer

class Optimer(gs_optimer.Optimer):

    def __init__(self, _config):
        super(Optimer, self).__init__(_config)

    def _setuplrparams(self, _params):
        _config = self.optimConfg
        _lparams = super(Optimer, self)._setuplrparams(_params)
        _lparams.append(
            {'params': [_params['mask']],
             'lr': _config.mask_lr, "name": "mask"}
        )
        return _lparams