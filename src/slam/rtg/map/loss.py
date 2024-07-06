from slam.rtg.common.map import imodel
from common.gs import points_utils
from utils import loss_utils
import torch
from torch.nn import functional as F

class LossWithPolicy(points_utils.ModuleBase):
    def __init__(self, _config) -> None:
        super(LossWithPolicy, self).__init__()
        self.config = _config
        self.optimConfg = _config.optim
        self.mapperConfg = _config.mapper

    def _attach_loss(self, _history, _state):
        _opacities = self.opacity_activation(_history["opacities"])
        _mask = (_opacities < 0.9).squeeze()

        if _mask.sum() > 0:
            _loss = 1000 * (
                loss_utils.l2_loss(
                    _state['scaling'][_mask],
                    _history["scaling"][_mask]
                )
                + loss_utils.l2_loss(
                    _state['xyz'][_mask],
                    _history["xyz"][_mask]
                )
                + loss_utils.l2_loss(
                    _state['rotation'][_mask],
                    _history["rotation_raw"][_mask]
                )
            )
        else:
            _loss = _mask.sum()

        return _loss

    def forward(self, _input, _renderout):
        _attach_loss = self._attach_loss(_input['state'], _renderout['state'])

        _color, _depth, _normal, _depth_index = (
            _renderout["color"], _renderout["depth"], _renderout["normal"], _renderout["depth_index_map"])

        _ssim_loss = torch.tensor(0, dtype= torch.float32, device = _color.device)
        _normal_loss = torch.tensor(0, dtype= torch.float32, device = _color.device)
        _depth_loss = torch.tensor(0, dtype= torch.float32, device = _color.device)

        _render_mask = _input['render_mask']

        if _render_mask is None:
            _render_mask = torch.ones(_color.shape[:2], dtype= torch.bool, device = _color.device)
            _ssim_loss = 1 - loss_utils.ssim(_color.permute(2,0,1), _input["color"].permute(2,0,1))
        else:
            _render_mask = _render_mask.bool()
        # render_mask include depth == 0
        if False: #self.dataset_type == "Scannetpp":
            _render_mask = _render_mask & (_input["depth"] > 0).squeeze()

        _color_loss = loss_utils.l1_loss(_color[_render_mask], _input["color"][_render_mask])

        if _depth is not None and self.optimConfg.depth_weight > 0:
            _depth_error = _depth - _input["depth"]
            _valid_depth_mask = (
                (_depth_index != -1).squeeze()
                & (_input["depth"] > 0).squeeze()
                & (_depth_error < self.mapperConfg.add_depth_thres).squeeze()
                & _render_mask
            )
            _depth_loss = torch.abs(_depth_error[_valid_depth_mask]).mean()

        if _normal is not None and self.optimConfg.normal_weight > 0:
            _cos_dist = 1 - F.cosine_similarity(
                _normal, _input["depth"], dim=-1
            )
            _valid_normal_mask = (
                _render_mask
                & (_depth_index != -1).squeeze()
                & (~(_input["depth"] == 0).all(dim=-1))
            )
            _normal_loss = _cos_dist[_valid_normal_mask].mean()

        _loss = _attach_loss + \
            self.optimConfg.depth_weight * _depth_loss + \
            self.optimConfg.normal_weight * _normal_loss + \
            self.optimConfg.color_weight * _color_loss + \
            self.optimConfg.ssim_weight * _ssim_loss

        _report_losses = {
            "loss": _loss.item(),
            "depth": _depth_loss.item(),
            "ssim": _ssim_loss.item(),
            "normal": _normal_loss.item(),
            "color": _color_loss.item(),
            "scale": _attach_loss.item(),
        }

        return _loss, _report_losses
