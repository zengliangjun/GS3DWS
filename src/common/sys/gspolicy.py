from common.sys import gsmodel
from abc import  ABC, abstractmethod

class IPolicy(ABC):

    def setup(self, _config):
        print("Do no thing!!!!!!")
        pass

    def preUpdate(self, _config):
        print("Do no thing!!!!!!")
        pass

    @abstractmethod
    def update(self, _iter: int, _model: gsmodel.IModule, _rout: dict):
        pass
