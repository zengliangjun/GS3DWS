#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization_monitor import GaussianRasterizationSettings, GaussianRasterizer

from utils.sh_utils import eval_sh
from common.gs import points_utils

class Render(points_utils.ModuleBase):
    def __init__(self, _config):
        self.config = _config
        self.renderConf = _config.render
        self.bg_color = torch.tensor(self.renderConf.bg_color, dtype= torch.float32)
        self.scaling_modifier = 1.0

    def __call__(self, _viewFrame, _parameras, _override_color = None) -> dict:
        _xyz = _parameras['xyz']
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        _2dpoints = torch.zeros_like(_xyz, requires_grad=True)
        try:
            _2dpoints.retain_grad()
        except:
            pass

        _tanfovx = math.tan(_viewFrame.fovX * 0.5)
        _tanfovy = math.tan(_viewFrame.fovY * 0.5)
        _device = _xyz.device

        _settings = GaussianRasterizationSettings(
            image_height=int(_viewFrame.height),
            image_width=int(_viewFrame.width),
            tanfovx=_tanfovx,
            tanfovy=_tanfovy,
            bg=self.bg_color.to(_device),
            scale_modifier=self.scaling_modifier,
            viewmatrix=_viewFrame.w2cTensor.transpose(0, 1).to(_device),  ## w2c
            projmatrix=_viewFrame.w2cViewTensor.transpose(0, 1).to(_device),  ## w2c
            sh_degree=_parameras['active_sh_degree'],
            campos= _viewFrame.c2wCenterTensor.to(_device),
            prefiltered=False,
            debug=self.renderConf.debug
        )

        _rasterizer = GaussianRasterizer(raster_settings = _settings)

        _means3D = _xyz
        _means2D = _2dpoints
        _opacities = _parameras['opacities']

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        _scaling = None
        _rotations = None
        _cov3D_precomp = None
        if self.renderConf.compute_cov3D_python:
            _cov3D_precomp = self.covariance_activation(_parameras['scaling'], self.scaling_modifier, _parameras['rotation'])
        else:
            _scaling = _parameras['scaling']
            _rotations = _parameras['rotation']

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        _shs = None
        _colors_precomp = None
        if _override_color is None:
            if self.renderConf.convert_SHs_python:
                shs_view = _parameras['features'].transpose(1, 2).view(-1, 3, (self.renderConf.sh_degree + 1)**2)
                dir_pp = (_xyz - _settings.campos.repeat(_xyz.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(_settings.sh_degree, shs_view, dir_pp_normalized)
                _colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                _shs = _parameras['features']
        else:
            _colors_precomp = _override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        _image, _depth, _median_depth, _alpha, _radii = _rasterizer(
            means3D = _means3D,
            means2D = _means2D,
            shs = _shs,
            colors_precomp = _colors_precomp,
            opacities = _opacities,
            scales = _scaling,
            rotations = _rotations,
            cov3D_precomp = _cov3D_precomp)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"color": _image,
                "depth": _depth,
                "2dradii": _radii,
                "uv": _2dpoints,
                "mask" : _radii > 0,
                "alpha" : _alpha,
                "median" : _median_depth,
                }
