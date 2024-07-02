import os.path as osp
import sys
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

_project_dir = osp.join(osp.dirname(__file__), "../../")
_project_dir = osp.abspath(_project_dir)

def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)

# src python path
add_pypath(osp.join(_project_dir, "src"))

from prune import config

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

from simple import gstrainer

@hydra.main(
    version_base=None,
    config_path=osp.join(_project_dir, "config/"),
    config_name="prune_compact3dgs"
)
def main(_cfg: DictConfig):
    OmegaConf.resolve(_cfg)
    #print(OmegaConf.to_yaml(_cfg))
    _config = config.load_compact3dgs_config(_cfg)

    _config.sys.dataCls = "gausiansplatting.data.nerf.NerfSource"
    _config.sys.sceneCls = "gausiansplatting.scene.gs_scene.Scene"
    _config.sys.modelCls = "prune.compcat.model.Model"
    _config.sys.policyCls = "prune.compcat.policy.Policy"
    _config.sys.optimerCls = "prune.compcat.optimer.Optimer"
    _config.sys.lossCls = "prune.compcat.loss.Loss"
    #_config.sys.renderCls = "gausiansplatting.scene.gs_render.Render"
    _config.sys.renderCls = "gausianmonitor.scene.monitor_render.Render"


    gstrainer.main(_config)


if __name__ == "__main__":
    main()
