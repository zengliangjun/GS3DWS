from common.sys import gsoptimer
import torch
import numpy as np

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

class Optimer(gsoptimer.IOptimer):

    def __init__(self, _config):
        self.config = _config
        self.optimConfg = _config.optim

    def _setuplrparams(self, _params):
        _config = self.optimConfg

        if hasattr(self.config, "nerf_normalization"):
            _spatial_lr_scale = self.config.nerf_normalization["radius"]
        else:
            _spatial_lr_scale = 3.9 ## spatial ## TODO

        _lparams = [
            {'params': [_params['xyz']],
             'lr': _config.position_lr_init * _spatial_lr_scale, "name": "xyz"},
            {'params': [_params['features_dc']],
             'lr': _config.feature_lr, "name": "features_dc"},
            {'params': [_params['features_rest']],
             'lr': _config.feature_lr / 20.0, "name": "features_rest"},
            {'params': [_params['opacities']],
             'lr': _config.opacity_lr, "name": "opacities"},
            {'params': [_params['scaling']],
             'lr': _config.scaling_lr, "name": "scaling"},
            {'params': [_params['rotation']],
             'lr': _config.rotation_lr, "name": "rotation"}
        ]
        return _lparams


    def setup(self, _params):
        _lparams = self._setuplrparams(_params)
        _config = self.optimConfg

        if hasattr(self.config, "nerf_normalization"):
            _spatial_lr_scale = self.config.nerf_normalization["radius"]
        else:
            _spatial_lr_scale = 3.9 ## spatial ## TODO

        self.optimizer = torch.optim.Adam(_lparams, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init= _config.position_lr_init * _spatial_lr_scale,
                                                    lr_final= _config.position_lr_final * _spatial_lr_scale,
                                                    lr_delay_mult= _config.position_lr_delay_mult,
                                                    max_steps= _config.position_lr_max_steps)

    def step(self, _iter):
        '''
        update learning rate
        '''
        ''' Learning rate scheduling per step '''
        for _group in self.optimizer.param_groups:
            if _group["name"] == "xyz":
                lr = self.xyz_scheduler_args(_iter)
                _group['lr'] = lr
                break

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)

    def update(self, _params):
        for _group in self.optimizer.param_groups:
            assert len(_group["params"]) == 1
            _name = _group["name"]
            _param = _params[_name]

            _orgParam = _group['params'][0]
            _group["params"][0] = _param

            _stored_state = self.optimizer.state.get(_orgParam, None)
            if (_stored_state is not None):
                if  (_orgParam.shape != _param.shape):
                    _exp_avg = torch.zeros_like(_param)
                    _exp_avg_sq = torch.zeros_like(_param)

                    _size = min(_orgParam.shape[0], _param.shape[0])

                    _org_exp_avg = _stored_state["exp_avg"]
                    _org_exp_avg_sq = _stored_state["exp_avg_sq"]

                    _exp_avg[: _size] = _org_exp_avg[: _size]
                    _exp_avg_sq[: _size] = _org_exp_avg_sq[: _size]

                    _stored_state["exp_avg"] = _exp_avg
                    _stored_state["exp_avg_sq"] = _exp_avg_sq

                del self.optimizer.state[_orgParam]
                self.optimizer.state[_param] = _stored_state

            del _orgParam

    def merger(self, _params):
        _optimizable_tensors = {}
        for _group in self.optimizer.param_groups:
            assert len(_group["params"]) == 1
            _extension_tensor = _params[_group["name"]]

            _stored_state = self.optimizer.state.get(_group['params'][0], None)
            if _stored_state is not None:

                _stored_state["exp_avg"] = torch.cat((_stored_state["exp_avg"], torch.zeros_like(_extension_tensor)), dim=0)
                _stored_state["exp_avg_sq"] = torch.cat((_stored_state["exp_avg_sq"], torch.zeros_like(_extension_tensor)), dim=0)

                del self.optimizer.state[_group['params'][0]]
                _group["params"][0] = nn.Parameter(torch.cat((_group["params"][0], _extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[_group['params'][0]] = _stored_state

                _optimizable_tensors[_group["name"]] = _group["params"][0]
            else:
                _group["params"][0] = nn.Parameter(torch.cat((_group["params"][0], _extension_tensor), dim=0).requires_grad_(True))
                _optimizable_tensors[_group["name"]] = _group["params"][0]

        return _optimizable_tensors

    def prune(self, _mask):
        _valid_mask = ~_mask

        _optimizable_tensors = {}
        for _group in self.optimizer.param_groups:
            _stored_state = self.optimizer.state.get(_group['params'][0], None)
            if _stored_state is not None:
                _stored_state["exp_avg"] = _stored_state["exp_avg"][_valid_mask]
                _stored_state["exp_avg_sq"] = _stored_state["exp_avg_sq"][_valid_mask]

                del self.optimizer.state[_group['params'][0]]
                _group["params"][0] = nn.Parameter((_group["params"][0][_valid_mask].requires_grad_(True)))
                self.optimizer.state[_group['params'][0]] = _stored_state

                _optimizable_tensors[_group["name"]] = _group["params"][0]
            else:
                _group["params"][0] = nn.Parameter(_group["params"][0][_valid_mask].requires_grad_(True))
                _optimizable_tensors[_group["name"]] = _group["params"][0]
        return _optimizable_tensors
