from gausiansplatting.scene import gs_scene
from surf2dgaussian.common import frame

class Scene(gs_scene.Scene):

    def __init__(self, _config):
        super(Scene, self).__init__(_config)

    def __next__(self):
        _frame = next(self.data)
        return frame.SurfFrame(_frame).to(self.device())
