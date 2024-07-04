from common.gs import frame
from utils import geometry_utils

import copy
import torch
import numpy as np

class Frame(frame.GSFrame):
    def __init__(self, _frame) -> None:
        super(Frame, self).__init__(_frame)

    def updatePoseC2W(self, _C2W):
        _w2c = np.linalg.inv(_C2W)
        self.w2cR = _w2c[:3, :3]
        self.w2cT = _w2c[:3, 3]
        self.w2c = _w2c

    def projectTensor(self, _xyzW):
        intrinsic = self.intrinsicTensor.to(_xyzW.device)
        w2c = self.w2cTensor.to(_xyzW.device)

        _xyzC = _xyzW @ w2c[:3, :3].T + w2c[:3, 3]
        _uv = _xyzC @ intrinsic.T
        _uv = _uv[:, :2] / _uv[:, 2:]
        _uv = _uv.long()
        return _uv

    def preTracking(self):
        self.trackColor = torch.tensor(self.data.color / 255, dtype = torch.float32).permute(2, 0, 1) # c*h*w
        if len(self.data.depth.shape) == 2:
            self.data.depth = self.data.depth[:, :, np.newaxis]

        self.trackDepth = torch.tensor(self.data.depth, dtype = torch.float32).permute(2, 0, 1)  # c*h*w


    def postTracking(self, _pose):
        self.updatePoseC2W(_pose)

    def preMapping(self, _config):
        _preprocessCfg = _config.mapper.preprocess


        _color = self.trackColor.permute(1, 2, 0) ## hxwxc
        _depth = self.trackDepth.permute(1, 2, 0) ## hxwxc     #### see frame.Frame

        _intrinsic = self.intrinsicTensor.to(_depth.device)

        if True:  # TODO flag for where is the best for the operator
            # depth filter
            if False: #_config.depth_filter:
                #_depth = utils.bilateralFilter(_depth, 5, 2, 2)
                ## TODO
                pass
            else:
                _depth = _depth

            _valid_mask = (_depth > _preprocessCfg.min_depth) & (_depth < _preprocessCfg.max_depth)
            _depth[~_valid_mask] = 0.0
            # update depth map
            self.trackDepth = _depth.permute(2, 0, 1)

        # compute geometry info
        _vertex_map = geometry_utils.vertex(_depth, _intrinsic)
        _normal_map = geometry_utils.normal(_vertex_map)
        _confidence_map = geometry_utils.confidence(_normal_map, _intrinsic)

        _invalid_mask = ((_normal_map == 0).all(dim=-1)) | \
            (_confidence_map < _preprocessCfg.invalid_confidence_thresh)[..., 0]

        _depth[_invalid_mask] = 0
        _vertex_map[_invalid_mask] = 0
        _normal_map[_invalid_mask] = 0
        #_confidence_map[_invalid_mask] = 0

        #####
        _c2w = self.c2wTensor.to(_depth.device)

        _vertex_w = geometry_utils.transform(_vertex_map, _c2w)
        _rotation = torch.eye(4, dtype = _c2w.dtype, device = _c2w.device)
        _rotation[:3, :3] = _c2w[:3, :3]
        _normal_w = geometry_utils.transform(_normal_map, _rotation)

        _init_params = {'color': _color,
                        'depth': _depth,
                        'vertex_w': _vertex_w,
                        'normal_w': _normal_w}
        setattr(self, 'params', _init_params)

    def to(self, _device):
        if hasattr(self, "trackColor"):
            self.trackColor = self.trackColor.to(_device)
        if hasattr(self, "trackDepth"):
            self.trackDepth = self.trackDepth.to(_device)

        if not hasattr(self, "params"):
            return

        _params= self.params
        for _key in _params:
            _params[_key] = _params[_key].to(_device)

    def cpuclone(self):
        if not hasattr(self, "params"):
            return
        assert self.params['color'].is_cuda == False
        return copy.deepcopy(self)

    def init_keys(self) -> list:
        return ['color', 'depth', 'vertex_w', 'normal_w']

    def optim_keys(self) -> list:
        return ['color', 'depth']

    def init_params(self) -> dict:
        assert hasattr(self, "params")
        return self.params

    def optim_params(self) -> dict:
        assert hasattr(self, "params")
        return self.params

    def toOptimStatus(self):

        if hasattr(self, "trackColor"):
            delattr(self, "trackColor")
        if hasattr(self, "trackDepth"):
            delattr(self, "trackDepth")

        assert hasattr(self, "params")
        _iKeys = self.init_keys()
        _oKeys = self.optim_keys()

        _delKeys = []
        for _key in _iKeys:
            if _key not in _oKeys:
                _delKeys.append(_key)

        _params = self.params
        for _key in _delKeys:
            _value = _params.pop(_key)
            del _value


