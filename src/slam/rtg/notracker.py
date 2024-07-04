from slam.common import itracker

class Tracker(itracker.Tracker):

    def __init__(self, _args):
        self.config = _args # template args MapPreProcessCfg
        self.trackPoses = []

    def tracking(self, _frameData) -> dict:
        _frameData.preTracking()
        self.trackPoses.append(_frameData.c2wGT)
        _frameData.postTracking(self.trackPoses[-1])
        return self.trackPoses
