from common.sys import gsmodel
from abc import  ABC, abstractmethod

class IPolicy(ABC):

    @abstractmethod
    def update(self, _iter: int, _model: gsmodel.IModule, _rout: dict):
        pass
