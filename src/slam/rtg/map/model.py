from slam.common.map import imodel
from slam.rtg.map import model_new, model_point, utils
import cuda_utils._C as cuda_utils

import torch
from torch import nn

class SLAMModel(imodel.SLAMModel):

    def __init__(self, _args):
        self.config = _args
        self.mapperConfg = _args.mapper
        self.newWarp = model_new.NewWarp(_args)

        self.pointcloud = model_point.MapClouds(_args)
        self.stablecloud = model_point.MapClouds(_args)

    def newFrame(self, _frameData):
        self.newWarp(_frameData)

    def render(self, _frameData, _type: imodel.PointType = imodel.PointType.STABLE_POINTS, _mask = None) -> dict:
        _renderParams = self.renderparams(_type)

        assert hasattr(self.config, "slam_render")
        _render = self.config.slam_render

        _routput = _render.render(_frameData, _renderParams, _mask)
        _output = {}

        def _permute(_key):
            return _routput[_key].permute(1, 2, 0)

        _output["color"] = _permute("color")
        _output["depth"] = _permute("depth")
        _output["normal"] = _permute("normal")
        _output["color_index_map"] = _permute("color_index_map")
        _output["depth_index_map"] = _permute("depth_index_map")
        _output["transmission"] = _permute("T_map")

        return _output

    def renderparams(self, _type: imodel.PointType = imodel.PointType.GLOBAL_POINTS) -> dict:
        if imodel.PointType.STABLE_POINTS == _type:
            _renderParams = self.stablecloud.renderparams()
        if imodel.PointType.UNSTABLE_POINTS == _type:
            _renderParams = self.pointcloud.renderparams()
        if imodel.PointType.GLOBAL_POINTS == _type:
            _sParams = self.stablecloud.renderparams()
            _uParams = self.pointcloud.renderparams()
            if _sParams is None and _uParams is None:
                return None
            if _sParams is None:
                return _uParams
            if _uParams is None:
                return _sParams

            _renderParams = {}
            for _key in _sParams:
                if not isinstance(_sParams[_key], torch.Tensor):
                        continue

                _renderParams[_key] = torch.cat([_sParams[_key], _uParams[_key]])
        return _renderParams

    def optim_params(self, _type: imodel.PointType = imodel.PointType.STABLE_POINTS) -> dict:
        if imodel.PointType.UNSTABLE_POINTS == _type:
            _gs = self.pointcloud.points
        if imodel.PointType.STABLE_POINTS == _type:
            _gs = self.stablecloud.points

        _gs.xyz = nn.Parameter(_gs.xyz.requires_grad_(True))
        _gs.features_dc = nn.Parameter(_gs.features_dc.requires_grad_(True))
        _gs.features_rest = nn.Parameter(_gs.features_rest.requires_grad_(True))
        _gs.scaling = nn.Parameter(_gs.scaling.requires_grad_(True))
        _gs.rotation = nn.Parameter(_gs.rotation.requires_grad_(True))
        _gs.opacities = nn.Parameter(_gs.opacities.requires_grad_(True))

        _params = {"xyz": _gs.xyz,
            "features_dc": _gs.features_dc,
            "features_rest": _gs.features_rest,
            "scaling": _gs.scaling,
            "rotation": _gs.rotation,
            "opacities": _gs.opacities}

        return _params

    def detach(self, _type: imodel.PointType = imodel.PointType.STABLE_POINTS):
        if imodel.PointType.UNSTABLE_POINTS == _type:
            _gs = self.pointcloud.points
        if imodel.PointType.STABLE_POINTS == _type:
            _gs = self.stablecloud.points

        _gs.xyz = _gs.xyz.detach()
        _gs.features_dc = _gs.features_dc.detach()
        _gs.features_rest = _gs.features_rest.detach()
        _gs.scaling = _gs.scaling.detach()
        _gs.rotation = _gs.rotation.detach()
        _gs.opacities = _gs.opacities.detach()

    def states(self, _type: imodel.PointType = imodel.PointType.STABLE_POINTS) -> dict:
        if imodel.PointType.UNSTABLE_POINTS == _type:
            _pointcloud =  self.pointcloud
        if imodel.PointType.STABLE_POINTS == _type:
            _pointcloud =  self.stablecloud

        _gs = _pointcloud.points
        _stat = {
            "xyz": _gs.xyz.detach().clone(),
            "features_dc": _gs.features_dc.detach().clone(),
            "features_rest": _gs.features_rest.detach().clone(),

            "opacities": _gs.opacities.detach().clone(),
            "confidences": _pointcloud.state.confidences.detach().clone(),
            "scaling": _gs.scaling.detach().clone(),
            "rotation": _pointcloud.rotation.detach().clone(),
            "rotation_raw": _gs.rotation.detach().clone(),
        }
        return _stat

    def merger(self, _params: dict, _type: imodel.PointType.UNSTABLE_POINTS):
        if imodel.PointType.UNSTABLE_POINTS == _type:
            _pointcloud =  self.pointcloud
        if imodel.PointType.STABLE_POINTS == _type:
            _pointcloud =  self.stablecloud
        _pointcloud.merger(_params)

    def merge_history(self, _state, _type: imodel.PointType = imodel.PointType.STABLE_POINTS):
        if imodel.PointType.UNSTABLE_POINTS == _type:
            _pointcloud =  self.pointcloud
        if imodel.PointType.STABLE_POINTS == _type:
            _pointcloud =  self.stablecloud

        _weight = self.mapperConfg.history_merge_max_weight
        if _weight <= 0:
            return

        _weight = _weight * _state["confidences"] / (_pointcloud.state.confidences + 1e-6)

        if False:
            print(f"_weight: {_weight.mean()}")

        _alpha = 1 - _weight

        _gs = _pointcloud.points
        _gs.xyz = _state["xyz"] * _weight + _alpha * _gs.xyz
        _gs.features_dc = _state["features_dc"] * _weight[0] + _alpha[0] * _gs.features_dc
        _gs.features_rest = _state["features_rest"] * _weight[0] + _alpha[0] * _gs.features_rest
        _gs.scaling = _state["scaling"] * _weight + _alpha * _gs.scaling
        _gs.rotation = utils.slerp(_state["rotation"], self.pointcloud.rotation, _alpha)


    def points_num(self, _type: imodel.PointType = imodel.PointType.STABLE_POINTS) -> int:
        if imodel.PointType.UNSTABLE_POINTS == _type:
            return self.pointcloud.points_num
        if imodel.PointType.STABLE_POINTS == _type:
            return self.stablecloud.points_num
        if imodel.PointType.GLOBAL_POINTS == _type:
            return self.stablecloud.points_num + self.pointcloud.points_num

    def intersect(self, _frameData, _xyz):
        _uv = _frameData.projectTensor(_xyz)
        _indices = torch.arange(_uv.shape[0]).cuda().long()
        _mask = (
            (_uv[:, 0] >= 0)
            & (_uv[:, 0] < _frameData.width)
            & (_uv[:, 1] >= 0)
            & (_uv[:, 1] < _frameData.height)
        )
        _valid_uv = _uv[_mask]

        # get the corresponding stable gaussians
        _render_output = self.render(_frameData, imodel.PointType.STABLE_POINTS)

        _rindex = _render_output["color_index_map"]
        _intersect_mask = _rindex[_valid_uv[:, 1], _valid_uv[:, 0]] >= 0
        _indices = _indices[_mask][_intersect_mask[:, 0]]

        # compute point to plane distance
        _rindex = (_rindex[_valid_uv[_indices, 1], _valid_uv[_indices, 0]]).squeeze(-1).long()

        _normal_check = self.stablecloud.normals[_rindex]
        _xyz_check = self.stablecloud.xyz[_rindex]

        _ixyz = _xyz[_indices]
        _distance = ((_xyz_check - _ixyz) * _normal_check).sum(dim=-1)
        _check = _distance.abs() < 0.5 * self.mapperConfg.add_depth_thres
        return _indices[_check]

    def purneWithTimeStamp(self, _time_stamp, _type):

        if _type == imodel.PointType.UNSTABLE_POINTS:
            _stables = False
            _points = self.pointcloud
        
        if _type == imodel.PointType.STABLE_POINTS:
            _stables = True
            _points = self.stablecloud

        if _points.points_num == 0:
            return

        _big_mask = _points.radius > (_points.radius.mean() * 10).squeeze()
        # isolated_gaussian_mask = self.gaussians_isolated(
        #     pointcloud.get_xyz, self.KNN_num, threshold
        # )
        if not _stables:
            _expire_mask = ((_time_stamp - _points.state.add_ticks) > self.mapperConfg.unstable_time_window).squeeze()
            _mask = (_big_mask | _expire_mask)
        else:
            # delete_mask = big_gaussian_mask | isolated_gaussian_mask
            _mask = _big_mask

        if False:
            print("threshold: big num: {_big_mask.sum()}, unstable num: {_unstable.sum()}")

        _points.prune(_mask)

    def stable_fix(self, _final = False):
        if _final:
            _mask = self.pointcloud.state.confidences > -1
        else:
            _mask = self.pointcloud.state.confidences > self.mapperConfg.stable_confidence_thres

        _mask = _mask.squeeze()


        if False:
            print(f"fix gaussian num: {_mask.sum()}")

        if _mask.sum() <= 0:
            return

        _stable_params = self.pointcloud.move(_mask)
        _stable_params["confidences"] = torch.clip(_stable_params["confidences"], max = self.mapperConfg.stable_confidence_thres)

        self.stablecloud.merger(_stable_params)

    def updateWithError(self, _kframe):
        _gsmap = self.render(_kframe, imodel.PointType.GLOBAL_POINTS)
        # [unstable, stable]
        assert hasattr(_kframe, "params")
        _params = _kframe.params

        _derror = torch.abs(_params['depth'] - _gsmap['depth'])
        _derror[_derror < 0] = 0
        _cerror = torch.abs(_params['color'] - _gsmap['color'])
        _cerror = torch.sum(_cerror, dim=-1, keepdim=True)

        _nerror = torch.zeros_like(_derror, dtype = _derror.dtype, device= _derror.device)
        _invalid_mask = (_params['depth'] == 0) | (_gsmap['depth_index_map'] == -1)
        _invalid_mask = _invalid_mask.squeeze()

        _derror[_invalid_mask] = 0
        _cerror[_params['depth'] == 0] = 0
        #_nerror[_invalid_mask] = 0
        H, W = _cerror.shape[:2]
        P = self.points_num(imodel.PointType.GLOBAL_POINTS)
        (
            gaussian_color_error,
            gaussian_depth_error,
            gaussian_normal_error,
            outlier_count,
        ) = cuda_utils.accumulate_gaussian_error(
            H,
            W,
            P,
            _cerror,
            _derror,
            _nerror,
            _gsmap['color_index_map'],
            _gsmap['depth_index_map'],
            self.mapperConfg.add_color_thres,
            self.mapperConfg.add_depth_thres,
            self.mapperConfg.add_normal_thres,
            True,
        )

        color_filter_thres = 2 * self.mapperConfg.add_color_thres
        depth_filter_thres = 2 * self.mapperConfg.add_depth_thres

        depth_delete_mask = (gaussian_depth_error > depth_filter_thres).squeeze()
        color_release_mask = (gaussian_color_error > color_filter_thres).squeeze()
        if False:
            print(
                "color outlier num: {color_release_mask).sum()}, depth outlier num: {(depth_delete_mask).sum()}"
            )

        color_release_mask_stable = color_release_mask[self.pointcloud.points_num:, ...]
        depth_delete_mask_stable = depth_delete_mask[self.pointcloud.points_num:, ...]

        self.stablecloud.counter_update(color_release_mask_stable, depth_delete_mask_stable)
        color_mask, depth_mask = self.stablecloud.counter_calcute()

        self.stablecloud.prune(depth_mask)
        self.stablecloud.reset(color_mask[~depth_mask], _kframe.id)


    def update(self, _type):
        if _type == imodel.PointType.UNSTABLE_POINTS:
            _points = self.pointcloud
        
        if _type == imodel.PointType.STABLE_POINTS:
            _points = self.stablecloud

        _points.updateCount()
