from common.raw import points as rpoints
from abc import  ABC, abstractmethod

class IScene(ABC):

    @abstractmethod
    def loadRawPoints(self) -> rpoints.RawPointsTensor:
        pass

    @abstractmethod
    def load(self) -> dict:
        pass

    @abstractmethod
    def save(self, _iter, _params):
        pass

    @abstractmethod
    def device() -> str:
        pass
