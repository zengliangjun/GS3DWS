from slam.common import imapper
from slam.common.map import imodel
import torch

class Mapper(imapper.Mapper):

    def __init__(self, _args):
        self.config = _args
        self.optimConfg = _args.optim

    @property
    def device(self) -> str:
        return torch.device(self.config.scene.device)

    def update(self, _poses, _frameData):
        assert hasattr(self.config, "slam_state")
        assert hasattr(self.config, "slam_model")
        assert hasattr(self.config, "slam_optimizer")
        _state = self.config.slam_state
        _model = self.config.slam_model
        _optimizer = self.config.slam_optimizer

        _state.update_poses(_poses)

        #########
        _frameData.to(self.device)

        _frameData.preMapping(self.config) ##  self.config.premapping
        _model.newFrame(_frameData)

        _frameData.to(torch.device("cpu"))
        _frameData.toOptimStatus()

        #########
        _state.add(_frameData)
        if 0 == _frameData.id or 0 == (_frameData.id + 1) % self.optimConfg.gaussian_update_frame:
            _keyframe = _state.keyframe(_frameData)
            ##
            ## TODO Scannetpp
            ##
            if not _keyframe or _model.points_num(imodel.PointType.STABLE_POINTS) <= 0:
                _optimizer.local_optimiz()
            else:
                _optimizer.global_optimiz(False)

            _model.purneWithTimeStamp(_frameData.id, _type = imodel.PointType.UNSTABLE_POINTS)

        #########
        _model.stable_fix(_final = False)
        if _model.points_num(imodel.PointType.STABLE_POINTS) > 0:
            # check error by backprojection
            _kframe = _state.getkeyframe(-1)
            _kframe.to(self.device)
            _model.updateWithError(_kframe)
            _kframe.to(torch.device("cpu"))

        _model.purneWithTimeStamp(_frameData.id, _type = imodel.PointType.STABLE_POINTS)

    def final(self):
        assert hasattr(self.config, "slam_model")
        assert hasattr(self.config, "slam_optimizer")
        _model = self.config.slam_model
        _optimizer = self.config.slam_optimizer

        _model.stable_fix(_final = True)
        _optimizer.global_optimiz(True)


