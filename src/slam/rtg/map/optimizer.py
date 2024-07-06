import torch
from tqdm import tqdm
import random
import numpy as np

from slam.rtg.common.map import ioptimizer, imodel
from slam.rtg.map import optim_utils, loss


STABLE_POINTS_WITHFINAL = imodel.PointType.STABLE_POINTS.value + 10


class Optimizer(ioptimizer.Optimizer):

    def __init__(self, _args):
        self.config = _args
        self.optimConfg = _args.optim
        self.loss = loss.LossWithPolicy(_args)

    @property
    def device(self) -> str:
        return torch.device(self.config.scene.device)

    def _optimizer(self, _type):
        assert hasattr(self.config, "slam_model")
        _model = self.config.slam_model

        _final = False
        if _type == STABLE_POINTS_WITHFINAL:
            _final = True
            _type = imodel.PointType.STABLE_POINTS

        _params = _model.optim_params(_type)

        _factors = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        if imodel.PointType.STABLE_POINTS == _type:
            if _final:
                _factors = [0.0000,
                            self.optimConfg.feature_lr_coef,
                            self.optimConfg.feature_lr_coef,
                            1.0,
                            self.optimConfg.scaling_lr_coef,
                            self.optimConfg.rotation_lr_coef]
            else:
                _factors = [0.0000,
                            0.1,
                            0.1,
                            0.1,
                            0.1,
                            0.1]

        _lparams = [
            {"params": [_params["xyz"]], "lr": self.optimConfg.position_lr * _factors[0], "name": "xyz"},
            {"params": [_params["features_dc"]], "lr": self.optimConfg.feature_lr* _factors[1], "name": "features_dc"},
            {"params": [_params["features_rest"]], "lr": self.optimConfg.feature_lr / 20* _factors[2], "name": "features_rest"},
            {"params": [_params["opacities"]], "lr": self.optimConfg.opacity_lr* _factors[3], "name": "opacities"},
            {"params": [_params["scaling"]], "lr": self.optimConfg.scaling_lr* _factors[4], "name": "scaling"},
            {"params": [_params["rotation"]], "lr": self.optimConfg.rotation_lr* _factors[5], "name": "rotation"},
        ]
        return torch.optim.Adam(_lparams, lr = 0.0, eps = 1e-15), _params


    def _prePareWithFrames(self, _frames, _type: imodel.PointType):
        assert hasattr(self.config, "slam_model")
        _model = self.config.slam_model

        _render_masks = []
        _tile_masks = []

        if imodel.PointType.UNSTABLE_POINTS == _type:
            _global_opt = False
            _sample_ratio = -1

        if imodel.PointType.STABLE_POINTS == _type:
            _global_opt = True
            _sample_ratio = 0.4

        if STABLE_POINTS_WITHFINAL == _type:
            _type = imodel.PointType.STABLE_POINTS
            _global_opt = True
            _sample_ratio = -1

        for _frame in _frames:
            _output = _model.render(_frame, _type)


            _frame.to(self.device)
            _render_mask, _tile_mask = \
                optim_utils.evaluate_render_range(_output, _frame, \
                                                  global_opt = _global_opt, \
                                                  sample_ratio = _sample_ratio)

            _frame.to(torch.device("cpu"))

            _render_masks.append(_render_mask)
            _tile_masks.append(_tile_mask)

        return {
            "frames": _frames,
            "rmasks": _render_masks,
            "tmasks": _tile_masks,
        }

    def _optimLoop(self, _optimFrames: dict, _optimInfos: dict):

        assert hasattr(self.config, "slam_model")
        _model = self.config.slam_model

        _frames = _optimFrames["frames"]
        _render_masks = _optimFrames["rmasks"] 
        _tile_masks = _optimFrames["tmasks"]

        _optimizer = _optimInfos['optimizer']
        _loopcount = _optimInfos['loopcount']
        _type = _optimInfos['type'] #imodel.PointType, 
        _history_state = _optimInfos['history_state']
        _state = _optimInfos['state']

        _selectCalback = _optimInfos['selectCalback']

        if imodel.PointType.STABLE_POINTS == _type:
            _desc = "globa: "
        if imodel.PointType.UNSTABLE_POINTS == _type:
            _desc = f"local_{_frames[-1].id}: "

        with tqdm(total = _loopcount, desc = _desc) as pbar:
            for _iter in range(_loopcount):
                '''
                '''
                _select_index = _selectCalback(_iter, len(_frames))

                _frame = _frames[_select_index]
                _render_mask = _render_masks[_select_index]
                _tile_mask = _tile_masks[_select_index]


                if hasattr(self.config, "debug") and self.config.debug:
                    import cv2
                    cv2.imshow("tile", _tile_mask.detach().cpu().numpy().astype(np.uint8) * 255)
                    cv2.imshow("render", _render_mask.detach().cpu().numpy().astype(np.uint8) * 255)

                _ouput = _model.render(_frame, imodel.PointType.GLOBAL_POINTS, _tile_mask)

                # compute loss
                _frame.to(self.device)
                '''
                init 
                
                '''
                _params = _frame.params

                _input = {"state": _history_state,
                          "render_mask": _render_mask,
                          "color": _params['color'],
                          "depth": _params['depth']}
                _ouput["state"] = _state

                _loss, _items = self.loss.forward(_input, _ouput)
                # 1
                _loss.backward()
                _optimizer.step()

                # 2
                _model.update(_type)

                # 3
                _optimizer.zero_grad(set_to_none=True)
                # 4
                _frame.to(torch.device("cpu"))

                ##  TODO logout
                pbar.set_postfix({"loss": "{0:1.5f}".format(_loss)})
                pbar.update(1)


    def local_optimiz(self):

        assert hasattr(self.config, "slam_model")
        _model = self.config.slam_model

        assert hasattr(self.config, "slam_state")
        _state = self.config.slam_state

        _type = imodel.PointType.UNSTABLE_POINTS
        _loopcount = self.optimConfg.gaussian_update_iter

        def _selectCalback(_iter, _len):
            if _iter > _loopcount / 2:
                _index = -1
            else:
                _index = random.randint(0, _len - 1)
            return _index

        _history_state = _model.states(_type)
        _optimizer, _params = self._optimizer(_type)

        _optimFrames = self._prePareWithFrames(_state.frames, _type)
        _optimInfos = {
            'type': _type,
            'optimizer': _optimizer,
            'loopcount': _loopcount,
            'history_state': _history_state,
            'state': _params,
            'selectCalback': _selectCalback
        }

        self._optimLoop(_optimFrames, _optimInfos)

        _model.detach(_type)
        _model.merge_history(_history_state, _type)

    '''
    
    
    '''
    def _select_keyframe(self, _final: bool):
        assert hasattr(self.config, "slam_state")
        _state = self.config.slam_state

        if _final:
            _select_keyframe_num = _state.keyframe_num
        else:
            _select_keyframe_num = min(self.optimConfg.global_keyframe_num, _state.keyframe_num)

        _random_sample = False

        if _random_sample:
            if _select_keyframe_num >= _state.keyframe_num:
                _select_indexs = list(range(0, _state.keyframe_num))
            else:
                _select_indexs = np.random.choice(
                    np.arange(1, min(_select_keyframe_num * 2, _state.keyframe_num)),
                    _select_keyframe_num -1, replace=False).tolist() + [0]
        else:
            _select_indexs = list(range(_select_keyframe_num))

        _select_indexs = [(i * -1) -1 for i in _select_indexs]

        _frames = []
        for _index in _select_indexs:
            _frames.append(_state.getkeyframe(_index))
        return _frames


    def global_optimiz(self, _final: bool):
        assert hasattr(self.config, "slam_model")
        _model = self.config.slam_model

        if _model.points_num(imodel.PointType.STABLE_POINTS) < 0:
            return

        assert hasattr(self.config, "slam_state")
        _state = self.config.slam_state

        _type = imodel.PointType.STABLE_POINTS
        _loopcount = self.optimConfg.gaussian_update_iter

        if _final:
            _loopcount = _state.keyframe_num * self.optimConfg.final_global_iter

        _frames = self._select_keyframe(_final)

        def _selectCalback(_iter, _len):
            if (_iter > _loopcount / 2) and (not _final):
                _index = -1
            else:
                _index = random.randint(0, _len - 1)
            return _index

        _history_state = _model.states(_type)

        _typeFlag = _type
        if _final:
            _typeFlag = STABLE_POINTS_WITHFINAL

        _optimizer, _params = self._optimizer(_typeFlag)

        _optimFrames = self._prePareWithFrames(_frames, _typeFlag)
        _optimInfos = {
            'type': _type,
            'optimizer': _optimizer,
            'loopcount': _loopcount,
            'history_state': _history_state,
            'state': _params,
            'selectCalback': _selectCalback
        }

        self._optimLoop(_optimFrames, _optimInfos)
        _model.detach(_type)
