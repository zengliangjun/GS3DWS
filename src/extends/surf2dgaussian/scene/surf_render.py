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
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from utils.sh_utils import eval_sh
from common.gs import points_utils

def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
    RS = points_utils.build_scaling_rotation(
        torch.cat([scaling * scaling_modifier, 
                   torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
    trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
    trans[:,:3,:3] = RS
    trans[:, 3,:3] = center
    trans[:, 3, 3] = 1
    return trans



class Render(points_utils.ModuleBase):
    def __init__(self, _config):
        self.config = _config
        self.renderConf = _config.render
        self.bg_color = torch.tensor(self.renderConf.bg_color, dtype= torch.float32)
        self.scaling_modifier = 1.0

        self.covariance_activation = build_covariance_from_scaling_rotation

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
            ## TODO
            ## TODO
            ## TODO
            pass        
        else:
            _scaling = _parameras['scaling']
            _rotations = _parameras['rotation']

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        _shs = None
        _colors_precomp = None
        if _override_color is None:
            if False: #self.renderConf.convert_SHs_python:
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
        _image, _radii, _allmap = _rasterizer(
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
        _rets = {"color": _image,
                "2dradii": _radii,
                "uv": _2dpoints,
                "mask" : _radii > 0,
                }


        # additional regularizations
        _ralpha = _allmap[1:2]

        # get normal map
        # transform normal from view space to world space
        _rnormal = _allmap[2:5]
        _rnormal = (_rnormal.permute(1, 2, 0) @ (_settings.viewmatrix[ :3, :3].T)).permute( 2, 0, 1)
        
        # get median depth map
        _depth_median = _allmap[5:6]
        _depth_median = torch.nan_to_num(_depth_median, 0, 0)

        # get expected depth map
        _depth_expected = _allmap[0:1]
        _depth_expected = (_depth_expected / _ralpha)
        _depth_expected = torch.nan_to_num(_depth_expected, 0, 0)
        
        # get depth distortion map
        _rdist = _allmap[6:7]

        # psedo surface attributes
        # surf depth is either median or expected by setting depth_ratio to 1 or 0
        # for bounded scene, use median depth, i.e., depth_ratio = 1; 
        # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
        _sdepth = _depth_expected * (1 - self.renderConf.depth_ratio) + (self.renderConf.depth_ratio) * _depth_median

        # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
        _snormal = depth_to_normal(_viewFrame, _sdepth)
        _snormal = _snormal.permute(2,0,1)
        # remember to multiply with accum_alpha since render_normal is unnormalized.
        _snormal = _snormal * (_ralpha).detach()


        _rets.update({
                'ralpha': _ralpha,
                'rnormal': _rnormal,
                'rdist': _rdist,
                'sdepth': _sdepth,
                'snormal': _snormal,
        })

        return _rets

import torch
import math

def depths_to_points(view, depthmap):
    _device = depthmap.device
    c2w = (view.w2cTensor.to(_device)).inverse()
    W, H = view.width, view.height
    fx = W / (2 * math.tan(view.fovX / 2.))
    fy = H / (2 * math.tan(view.fovY / 2.))
    intrins = torch.tensor(
        [[fx, 0., W/2.],
        [0., fy, H/2.],
        [0., 0., 1.0]]
    ).float().cuda()
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device=_device).float(), torch.arange(H, device=_device).float(), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    rays_o = c2w[:3,3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points

def depth_to_normal(view, depth):
    """
        view: view camera
        depth: depthmap 
    """
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output
