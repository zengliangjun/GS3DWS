from collections import deque
from utils import geometry_utils
from slam.rtg.common.map import istate
from slam.rtg.common import iframe

class Manager(istate.StateManager):

    def __init__(self, _args):
        self.config = _args
        self.stateConfg = _args.mapper.state
        self.frames = deque(maxlen=self.stateConfg.memory_length)
        self.keyframes = []

    def update_poses(self, _poses: list):
        if _poses is None:
            return
        for _frame in self.frames:
            _frame.updatePoseC2W(_poses[_frame.id])

        for _frame in self.keyframes:
            _frame.updatePoseC2W(_poses[_frame.id])

    def add(self, _frame: iframe.Frame):
        self.frames.append(_frame)

    # check if current frame is a keyframe
    def keyframe(self, _frame: iframe.Frame) -> bool:
        # add keyframe
        if _frame.id == 0:
            self.keyframes.append(_frame.cpuclone())
            return False

        ####
        _prev_rot = self.keyframes[-1].w2cR.T  ####
        _prev_trans = self.keyframes[-1].w2cT
        _rot = _frame.w2cR.T  ####
        _trans = _frame.w2cT
        _, theta_diff = geometry_utils.compare_rotmat(_prev_rot, _rot)
        _, l2_diff = geometry_utils.compare_trans(_prev_trans, _trans)

        if theta_diff > self.stateConfg.keyframe_theta_thes or l2_diff > self.stateConfg.keyframe_trans_thes:
            self.keyframes.append(_frame.cpuclone())
            return True
        else:
            return False

    def getkeyframe(self, _id) -> iframe.Frame:
        return self.keyframes[_id]

    def __len__(self):
        return len(self.keyframes)

    @property
    def keyframe_num(self):
        return len(self.keyframes)

