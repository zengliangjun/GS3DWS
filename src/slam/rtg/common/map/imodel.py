import abc

import enum

class PointType(enum.Enum):
    STABLE_POINTS = 1
    UNSTABLE_POINTS = 2
    GLOBAL_POINTS = 3

class SLAMModel:

    @abc.abstractmethod
    def newFrame(self, _frameData):
        pass

    @abc.abstractmethod
    def intersect(self, _frameData, _xyz):
        pass


    @abc.abstractmethod
    def render(self, _type: PointType = PointType.STABLE_POINTS) -> dict:
        pass

    @abc.abstractmethod
    def renderparams(self, _type: PointType = PointType.GLOBAL_POINTS) -> int:
        pass

    @abc.abstractmethod
    def states(self, _type: PointType = PointType.STABLE_POINTS) -> dict:
        pass

    @abc.abstractmethod
    def merge_history(self, _state, _type: PointType = PointType.STABLE_POINTS):
        pass


    @abc.abstractmethod
    def stable_fix(self, _final = False):
        pass


    @abc.abstractmethod
    def buildWithId(_points, _time_stamp):
        pass

    @abc.abstractmethod
    def merger(self, _params: dict, _type: PointType = PointType.UNSTABLE_POINTS):
        pass

    @abc.abstractmethod
    def purneWithTimeStamp(self, _time_stamp, _type: PointType = PointType.STABLE_POINTS):
        pass

    @abc.abstractmethod
    def updateWithError(self, _kframe):
        pass

    @abc.abstractmethod
    def update(self, _type):
        pass

    @abc.abstractmethod
    def optim_params(self, _type: PointType = PointType.STABLE_POINTS) -> dict:
        pass

    @abc.abstractmethod
    def detach(self, _type: PointType = PointType.STABLE_POINTS):
        pass


    @abc.abstractmethod
    def points_num(self, _type: PointType = PointType.STABLE_POINTS) -> int:
        pass
