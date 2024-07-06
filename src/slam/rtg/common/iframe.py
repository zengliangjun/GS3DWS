import abc

class Frame:


    @abc.abstractmethod
    def preTracking(self):
        pass

    @abc.abstractmethod
    def postTracking(self):
        pass


    @abc.abstractmethod
    def preMapping(self):
        pass

    @abc.abstractmethod
    def init_params(self) -> dict:
        pass

    @abc.abstractmethod
    def optim_params(self) -> dict:
        pass

    @abc.abstractmethod
    def toOptimStatus(self):
        pass
