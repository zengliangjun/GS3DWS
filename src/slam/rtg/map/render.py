from slam.rtg.common import iframe
from common.gs import points_utils

from diff_gaussian_rasterization_depth import  GaussianRasterizationSettings, GaussianRasterizer
import math
import torch

class Render(points_utils.ModuleBase):

    def __init__(self, _args):
        super(Render, self).__init__()
        self.renderConf = _args.render
        self.bg_color = torch.tensor(self.renderConf.bg_color, dtype= torch.float32)
        self.scaling_modifier = 1

        self.config = _args

    def render(self, _frame: iframe.Frame, \
               _gaussian_data: dict, \
               _tile_mask = None) -> dict:

        _tanfovx = math.tan(_frame.fovX * 0.5)
        _tanfovy = math.tan(_frame.fovY * 0.5)

        _device = _gaussian_data['xyz'].device


        self.raster_settings = GaussianRasterizationSettings(
            image_height = int(_frame.height),
            image_width = int(_frame.width),
            tanfovx = _tanfovx,
            tanfovy = _tanfovy,
            bg = self.bg_color.to(_device),
            scale_modifier = self.scaling_modifier,
            viewmatrix = _frame.w2cTensor.transpose(0, 1).to(_device),  ## w2c
            projmatrix = _frame.w2cViewTensor.transpose(0, 1).to(_device),  ## w2c
            sh_degree = self.renderConf.active_sh_degree,
            campos = _frame.c2wCenterTensor.to(_device),
            opaque_threshold = self.renderConf.renderer_opaque_threshold,
            depth_threshold = self.renderConf.renderer_depth_threshold,
            normal_threshold = self.renderConf.renderer_normal_threshold,
            color_sigma = self.renderConf.color_sigma,
            prefiltered = False,
            debug = False,
            cx = _frame.cx,
            cy = _frame.cy,
            T_threshold = 0.0001,
        )

        self.rasterizer = GaussianRasterizer(raster_settings=self.raster_settings)

        means3D = _gaussian_data["xyz"]
        opacity = _gaussian_data["opacities"]
        scales = _gaussian_data["scaling"]
        rotations = _gaussian_data["rotation"]
        shs = _gaussian_data["features"]
        normal = _gaussian_data["normals"]
        cov3D_precomp = None
        colors_precomp = None
        if _tile_mask is None:
            _tile_mask = torch.ones(((_frame.height + 15) // 16, (_frame.width + 15) // 16), \
                                   dtype = torch.int32, device=means3D.device)

        _results = self.rasterizer(
            means3D=means3D,
            opacities=opacity,
            shs=shs,
            colors_precomp=colors_precomp,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
            normal_w=normal,
            tile_mask=_tile_mask,
        )

        rendered_image = _results[0]
        rendered_depth = _results[1]
        color_index_map = _results[2]
        depth_index_map = _results[3]
        color_hit_weight = _results[4]
        depth_hit_weight = _results[5]
        T_map = _results[6]

        render_normal = torch.zeros_like(rendered_image) #, device= rendered_image.device) , dtype= torch.float32
        render_normal[:, depth_index_map[0] > -1] = normal[depth_index_map[depth_index_map > -1].long()].permute(1, 0)

        results = {
            "color": rendered_image,
            "depth": rendered_depth,
            "normal": render_normal,
            "color_index_map": color_index_map,
            "depth_index_map": depth_index_map,
            "color_hit_weight": color_hit_weight,
            "depth_hit_weight": depth_hit_weight,
            "T_map": T_map,
        }
        if hasattr(self.config, "debug") and self.config.debug:
            import numpy as np
            import cv2
            _rgb = results["color"] * 255
            _rgb = _rgb.permute(1, 2, 0)
            _rgb = _rgb.detach().cpu().numpy().astype(np.uint8)
            cv2.imshow("color", _rgb)
            cv2.waitKey(0)

        return results

    def __call__(self, _viewFrame, _parameras, _override_color = None) -> dict:
        return self.render(_viewFrame, _parameras)
