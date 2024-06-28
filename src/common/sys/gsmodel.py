from common.raw import points
from abc import  ABC, abstractmethod

class IModule(ABC):

    @abstractmethod
    def renderparams(self):
        pass

    @abstractmethod
    def params(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def prune(self, _mask):
        '''
        prune gs point
        '''
        pass

    @abstractmethod
    def split(self, _mask):
        '''
        split gs point
        '''
        pass

    @abstractmethod
    def clone(self, _mask):
        '''
        prune gs point
        '''
        pass

    @abstractmethod
    def move(self, _mask):
        '''
        prune and return pruned data, dict
        '''
        pass

    @abstractmethod
    def merger(self, _paramters):
        '''
        prune and return pruned data, dict
        '''
        pass

    @abstractmethod
    def reset(self, _keyAttr: str):
        '''
        prune and return pruned data, dict
        '''
        pass


    @abstractmethod
    def build(self, _points: points.RawPointsTensor):
        '''
        Can't update module
        '''
        pass

    @abstractmethod
    def updateCount(self, _rout:dict):
        pass

    @abstractmethod
    def update(self, _paramters):
        pass

    @abstractmethod
    def save(self, _path: str, _params: dict = None):
        pass