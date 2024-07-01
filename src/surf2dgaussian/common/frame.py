from common.gs import frame as gsframe
from common.raw import frame
import torch

class SurfFrame(gsframe.GSFrame):

    def __init__(self, _frame: frame.Frame) -> None:
        super(SurfFrame, self).__init__(_frame)

    def to(self, _device):
        if not hasattr(self, "colorTensor"):
            self.colorTensor = torch.tensor(self.color[..., :3] / 255, dtype = torch.float32).permute(2, 0, 1) # c*h*w
            if 4 == self.color.shape[-1]:
                self.alphaTensor = torch.tensor(self.color[..., 3:], dtype = torch.float32).permute(2, 0, 1) # c*h*w

        self.colorTensor = self.colorTensor.to(_device)
        if hasattr(self, "alphaTensor") and self.alphaTensor is not None:
            self.alphaTensor = self.alphaTensor.to(_device)

        return self

