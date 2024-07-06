import abc
from slam.rtg.common import iframe

class StateManager():

    @abc.abstractmethod
    def update(self, _poses: list):
        pass

    @abc.abstractmethod
    def add(self, _frame: iframe.Frame):
        pass

    @abc.abstractmethod
    def keyframe(self, _frame: iframe.Frame) -> bool:
        pass

    @abc.abstractmethod
    def getkeyframe(self, _id) -> iframe.Frame:
        pass

