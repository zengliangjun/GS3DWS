from abc import  ABC, abstractmethod

class IOptimer(ABC):

    @abstractmethod
    def setup(self, _params):
        pass

    @abstractmethod
    def step(self, _iter):
        '''
        update learning rate
        '''
        pass

    @abstractmethod
    def update(self, _params):
        pass

    @abstractmethod
    def merger(self, _params):
        pass

    @abstractmethod
    def prune(self, _mask):
        '''
        prune gs point
        '''
        pass