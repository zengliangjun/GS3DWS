import abc
from slam.common import iframe

class Optimizer:

    @abc.abstractmethod
    def local_optimiz(self):
        pass

    @abc.abstractmethod
    def global_optimiz(self, _final: bool):
        pass
