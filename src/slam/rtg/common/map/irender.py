import abc
from slam.rtg.common import iframe

class Render:

    @abc.abstractmethod
    def render(self, _frame: iframe.Frame, \
               _gsdata: dict, \
               _tile_mask = None) -> dict:
        pass
