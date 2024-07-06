import abc
from slam.rtg.common import iframe

class Optimizer:

    @abc.abstractmethod
    def local_optimiz(self):
        pass

    @abc.abstractmethod
    def global_optimiz(self, _final: bool):
        pass
