
import torch
from pytorch3d import ops as p3dops

import RTG_simple_knn._C as sknn

from slam.rtg.map import model_point
from slam.rtg.common.map import imodel
from common.raw import points

class NewWarp:
    def __init__(self, _args):
        self.config = _args
        self.mapConfg = _args.mapper
        self.gaussianConfg = _args.mapper.gaussian
        self.newcloud = model_point.MapClouds(_args)
        self.debugInfo = {}

    def _sample_pixels(self, _params, _sample_num, _select_mask = None):
        assert _sample_num >= 0
        if _sample_num == 0:
            return points.RawPointsTensor()

        _color = _params["color"]
        _vertex = _params["vertex_w"]
        _normal = _params["normal_w"]
        # 1. compute local data

        _H, _W = _vertex.shape[0], _vertex.shape[1]
        _coord_y, _coord_x = torch.meshgrid(
            torch.arange(_H, device =_color.device), torch.arange(_W, device =_color.device), indexing="ij"
        )
        _coord_y = _coord_y.flatten()
        _coord_x = _coord_x.flatten()

        if _select_mask is None:
            _select_mask = torch.ones([_H, _W, 1], device = _vertex.device, dtype= torch.bool)

        _invalid_normal_mask = torch.where(_normal.sum(dim=-1) == 0)
        _select_mask[_invalid_normal_mask] = False
        if _sample_num > _select_mask.sum():
            _sample_num = _select_mask.sum()

        _select_mask = _select_mask.flatten()
        _vertex = _vertex.view(-1, 3)[_select_mask]
        _normal = _normal.view(-1, 3)[_select_mask]
        _color = _color.view(-1, 3)[_select_mask]

        _samples = torch.randperm(_vertex.shape[0])[:_sample_num]

        _vertex = _vertex[_samples]
        _normal = _normal[_samples]
        _color = _color[_samples]

        _points = points.RawPointsTensor()
        _points.colors = _color
        _points.normals = _normal
        _points.points = _vertex
        return _points

    def _initFrame(self, _frameData):
        _params = _frameData.init_params()
        _mask = _params["depth"] > 0
        _points = self._sample_pixels(_params, self.mapConfg.uniform_sample_num, _mask)
        _points_params = self.newcloud.buildWithId(_points, _frameData.id)
        self.newcloud.merger(_points_params)

    def _initFrameWithGsmap(self, _frameData, _gsout):
        _params = _frameData.init_params()

        _tmask = (_gsout["transmission"] > self.mapConfg.add_transmission_thres) & \
                                    (_params["depth"] > 0)
        _ratio = _tmask.sum() / _frameData.pixel_num

        _sample_num = int(self.mapConfg.transmission_sample_ratio * _ratio * self.mapConfg.uniform_sample_num)

        self.debugInfo['tmask'] = _tmask.sum().item()
        self.debugInfo['tsample'] = _sample_num

        _points = self._sample_pixels(_params, _sample_num, _tmask)
        _points_params = self.newcloud.buildWithId(_points, _frameData.id)
        self.newcloud.merger(_points_params)

        _derror = torch.abs(_params["depth"] - _gsout["depth"])
        _cerror = torch.abs(_params["color"] - _gsout["color"]).mean(dim = -1, keepdim = True)

        _d_mask = ((_derror > self.mapConfg.add_depth_thres)
                & (_params["depth"] > 0)
                & (_gsout["depth_index_map"] > -1))

        _c_mask = ((_cerror > self.mapConfg.add_color_thres)
                & (_params["depth"] > 0)
                & (_gsout["transmission"] < self.mapConfg.add_transmission_thres))

        _mask = _c_mask | _d_mask
        _mask = _mask & (~_tmask)
        _sample_num = int(_mask.sum() * self.mapConfg.error_sample_ratio)
        if False:
            print(f"wrong depth num = {_d_mask.sum()}, wrong color num = {_c_mask.sum()}, sample num = {_sample_num}")

        self.debugInfo['dmask'] = _d_mask.sum().item()
        self.debugInfo['cmask'] = _c_mask.sum().item()
        self.debugInfo['dcmask'] = _sample_num


        _points = self._sample_pixels(_params, _sample_num, _mask)
        _points_params = self.newcloud.buildWithId(_points, _frameData.id)
        self.newcloud.merger(_points_params)

    def _init(self, _frameData):
        if _frameData.id == 0:
            self._initFrame(_frameData)
        else:
            assert hasattr(self.config, "slam_model")
            _model = self.config.slam_model
            _gsout = _model.render(_frameData, imodel.PointType.GLOBAL_POINTS)
            self._initFrameWithGsmap(_frameData, _gsout)


    def _bbox_filter(self, _exist_xyz, padding=0.05):
        _xyz = self.newcloud.xyz

        _min = _xyz.min(dim=0)[0] - padding
        _max = _xyz.max(dim=0)[0] + padding
        _mask = (_exist_xyz > _min).all(dim=-1) & (_exist_xyz < _max).all(dim=-1)
        return _mask

    def _filter(self, _exist_params, _topk = 3):
        _exist_xyz = _exist_params["xyz"]
        _exist_raidus = _exist_params["radius"]

        #if torch.numel(_exist_xyz) > 0:
        _mask = self._bbox_filter(_exist_xyz)
        if _mask.sum() == 0:
            return
        _exist_xyz = _exist_xyz[_mask]
        _exist_raidus = _exist_raidus[_mask]


        _nn_dist, _nn_indices, _ = p3dops.knn_points(
            self.newcloud.xyz[None, ...],
            _exist_xyz[None, ...],
            norm=2,
            K= _topk,
            return_nn=True,
        )
        _nn_dist = torch.sqrt(_nn_dist).squeeze(0)
        _nn_indices = _nn_indices.squeeze(0)

        _corr_radius = _exist_raidus[_nn_indices] * 0.6
        _inside_mask = (_nn_dist < _corr_radius).any(dim=-1)

        self.debugInfo['filter'] = _inside_mask.sum().item()

        self.newcloud.prune(_inside_mask)

    def _opacity_intersect(self, _frameData, intersect_calback, _opacity_low = 0.1):
        _xyz = self.newcloud.xyz
        _origin_indices = torch.arange(_xyz.shape[0]).cuda().long()

        _opacities = self.newcloud.opacities
        _opacity_filter = (_opacities > _opacity_low).squeeze(-1)

        _xyz = _xyz[_opacity_filter] ###

        _indices = intersect_calback(_frameData, _xyz)

        _indices = _origin_indices[_opacity_filter][_indices]

        # set opacity
        self.newcloud.points.opacities[_indices] = self.newcloud.points.opacity_inverse_activation(
            _opacity_low
            * torch.ones_like(self.newcloud.points.opacities[_indices])
        )

    def _update_geometry(self, _params):
        _points_num = self.newcloud.points_num

        _xyz = self.newcloud.xyz
        _radius = self.newcloud.radius

        if _params is not None:
            _extra_xyz = _params["xyz"]
            _extra_radius = _params["radius"]

            if torch.numel(_extra_xyz) > 0:
                _inbbox_mask = self._bbox_filter(_extra_xyz)
                _extra_xyz = _extra_xyz[_inbbox_mask]
                _extra_radius = _extra_radius[_inbbox_mask]

            _total_xyz = torch.cat([_xyz, _extra_xyz])
            _total_radius = torch.cat([_radius, _extra_radius])
        else:
            #_total_xyz = copy.deepcopy(_xyz)
            #_total_radius = copy.deepcopy(_radius)
            _total_xyz = _xyz
            _total_radius = _radius

        _, _knn_indices = sknn.distCUDA2(_total_xyz.float())
        _knn_indices = _knn_indices[:_points_num].long()

        dist_0 = (
            torch.norm(_xyz - _total_xyz[_knn_indices[:, 0]], p=2, dim=1)
            - 3 * _total_radius[_knn_indices[:, 0]]
        )
        dist_1 = (
            torch.norm(_xyz - _total_xyz[_knn_indices[:, 1]], p=2, dim=1)
            - 3 * _total_radius[_knn_indices[:, 1]]
        )
        dist_2 = (
            torch.norm(_xyz - _total_xyz[_knn_indices[:, 2]], p=2, dim=1)
            - 3 * _total_radius[_knn_indices[:, 2]]
        )
        invalid_dist_0 = dist_0 < 0
        invalid_dist_1 = dist_1 < 0
        invalid_dist_2 = dist_2 < 0

        invalid_scale_mask = invalid_dist_0 | invalid_dist_1 | invalid_dist_2
        dist2 = (dist_0 ** 2 + dist_1 ** 2 + dist_2 ** 2) / 3
        scales = torch.sqrt(dist2)
        scales = torch.clip(scales, min = self.gaussianConfg.min_radius, max = self.gaussianConfg.max_radius)

        self.newcloud.prune(invalid_scale_mask)

        _valid = ~invalid_scale_mask
        scales = scales[_valid]

        if _valid.sum() != 0:
            scales = scales[..., None].repeat(1, 3)
            _factor = torch.tensor(self.gaussianConfg.xyz_factor, dtype = scales.dtype, device= scales.device)
            _factor_scales = self.gaussianConfg.scale_factor * torch.mul(scales, _factor)
            self.newcloud.points.scaling = self.newcloud.points.scaling_inverse_activation(_factor_scales)

    def __call__(self, _frameData):
        ##### 1
        self._init(_frameData)

        assert hasattr(self.config, "slam_model")
        _model = self.config.slam_model
        if True:
            ##### 2
            _points_num = _model.points_num(imodel.PointType.UNSTABLE_POINTS)

            if 0 != _points_num:
                _exist_params = _model.renderparams(imodel.PointType.UNSTABLE_POINTS)
                self._filter(_exist_params)

        if True:
            ##### 3
            _points_num = _model.points_num(imodel.PointType.STABLE_POINTS)
            if 0 != _points_num:
                self._opacity_intersect(_frameData, _model.intersect)

        ##### merger
        _exist_params = _model.renderparams(imodel.PointType.GLOBAL_POINTS)
        self._update_geometry(_exist_params)

        ####
        self.debugInfo['merger'] = self.newcloud.points_num
        assert hasattr(self.config, "writer")
        self.config.writer.write_dict(self.debugInfo, _frameData.id)
        ####

        _mask = torch.ones(self.newcloud.points_num, \
                           dtype = torch.bool, \
                           device = self.newcloud.xyz.device)
        _params = self.newcloud.move(_mask)

        _model.merger(_params, imodel.PointType.UNSTABLE_POINTS)

