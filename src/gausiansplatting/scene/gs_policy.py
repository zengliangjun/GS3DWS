from common.sys import gspolicy, gsmodel, gsoptimer
import torch


class Policy(gspolicy.IPolicy):

    def __init__(self, _config) -> None:
        self.config = _config
        self.policyConfg = _config.policy

        self.white_background = True
        for _item in _config.scene.scene_color:
            self.white_background = self.white_background and _item == 255

    def setup(self, _config):
        pass

    def preUpdate(self, _config):
        pass

    def _clone_split(self, _model: gsmodel.IModule, _optimer: gsoptimer.IOptimer):
        _state = _model.state
        _grads = _state.xyz_gradient_accum / _state.confidences
        _grads[_grads.isnan()] = 0.0

        _select_mask = torch.where(torch.norm(_grads, dim=-1) >= self.policyConfg.densify_grad_threshold, True, False)

        if hasattr(self.config, "nerf_normalization"):
            _scene_extent = self.config.nerf_normalization["radius"]
        else:
            _scene_extent = 4 ## TODO

        _max = torch.max(_model.scaling, dim = 1).values
        _clone_mask = _max <= self.policyConfg.percent_dense * _scene_extent
        _split_mask = ~_clone_mask
        ###
        _clone_mask = torch.logical_and(_select_mask, _clone_mask)
        _split_mask = torch.logical_and(_select_mask, _split_mask)

        _colone_params = _model.clone(_clone_mask)
        _split_params = _model.split(_split_mask)

        _params = {}
        for _k in _colone_params:
            _params[_k] = torch.cat((_colone_params[_k], _split_params[_k]), dim = 0)
        
        _model.merger(_params)
        _params = _model.params()
        _optimer.update(_params)

        _count = _params['xyz'].shape[0]
        _prune_filter = torch.zeros(_count, dtype = _select_mask.dtype, device = _select_mask.device)
        _prune_filter[: _split_mask.shape[0]] = _split_mask
        return _prune_filter

    def _prune(self, _iter, _model: gsmodel.IModule):
        _opacities = _model.opacities
        _min_opacity_threshold = 0.005

        _prune_mask = (_opacities < _min_opacity_threshold).squeeze()

        if _iter > self.policyConfg.opacity_reset_interval:
            _size_threshold = 20

            _state = _model.state

            if hasattr(self.config, "nerf_normalization"):
                _scene_extent = self.config.nerf_normalization["radius"]
            else:
                _scene_extent = 4 ## TODO

            _big_points_vs = _state.max_2dradii > _size_threshold
            _scaling = _model.scaling
            _big_points_ws = torch.max(_scaling, dim =1).values > 0.1 * _scene_extent

            _prune_mask = torch.logical_or(torch.logical_or(_prune_mask, _big_points_vs), _big_points_ws)
        return _prune_mask

    def update(self, _iter: int, _model: gsmodel.IModule, _optimer: gsoptimer.IOptimer, _rout: dict):

        if _iter % 1000 == 0:
            _model.oneupSHdegree()

        # Densification
        if _iter > self.policyConfg.densify_until_iter:
            return

        # Keep track of max radii in image-space for pruning
        _model.updateCount(_rout)

        if _iter > self.policyConfg.densify_from_iter and _iter % self.policyConfg.densification_interval == 0:
            _prune_filter = self._clone_split(_model, _optimer)
            ########################################################
            _prune_mask = self._prune(_iter, _model)
            _prune_mask = torch.logical_or(_prune_filter, _prune_mask)
            _model.prune(_prune_mask)
            _params = _model.params()
            _optimer.update(_params)

        if _iter % self.policyConfg.opacity_reset_interval == 0 or \
            (self.white_background and _iter == self.policyConfg.densify_from_iter):
            _model.reset("opacities")
            _params = _model.params()
            _optimer.update(_params)

        torch.cuda.empty_cache()
