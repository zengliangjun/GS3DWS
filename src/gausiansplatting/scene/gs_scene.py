from common.raw import points as rpoints
from common.gs import points
from common.sys import gsscene, gsmodel
import hydra
import os
import os.path as osp
from common.gs import frame
import torch

class Scene(gsscene.IScene):

    def __init__(self, _config) -> None:
        self.sceneConfig = _config.scene
        _dataCls = hydra.utils.get_class(_config.sys.dataCls)
        self.data = _dataCls(_config)
        ### additional info
        setattr(_config, "nerf_normalization", self.data.nerf_normalization)

    def loadRawPoints(self) -> rpoints.RawPointsTensor:
        return self.data.loadRawPoints().tensor().to(self.device())

    def load(self) -> dict:
        _path = osp.join(self.sceneConfig.out_path, "point_cloud")
        if not osp.exists(_path):
            return None
        _iters = [int(_fname.split("_")[-1]) for _fname in os.listdir(_path)]
        if 0 == len(_iters):
            return None
        _iter = max(_iters)  ## TODO
        _path = osp.join(_path, f"iteration_{_iter}")
        _items = points.GSPoints.load(_path, self.sceneConfig.sh_degree)
        for _key in _items:
            _items[_key] = _items[_key].to(self.device)

        return _items

    def save(self, _iter, _gsmodel: gsmodel.IModule):
        if _iter not in self.sceneConfig.saving_iterations:
                return
        _path = osp.join(self.sceneConfig.out_path, "point_cloud/iteration_{}".format(_iter))
        os.makedirs(_path, exist_ok= True)
        _file = osp.join(_path, "point_cloud.ply")

        _params = {'iteration': _iter}
        _gsmodel.save(_file, _params)

    def device(self) -> str:
        return torch.device(self.sceneConfig.device)

    def __iter__(self):
        return self

    def __next__(self):
        _frame = next(self.data)
        return frame.GSFrame(_frame).to(self.device())
