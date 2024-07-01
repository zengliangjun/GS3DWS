import os.path as osp
import sys
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

_project_dir = osp.join(osp.dirname(__file__), "../")
_project_dir = osp.abspath(_project_dir)

def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)

# src python path
add_pypath(osp.join(_project_dir, "src"))

from surf2dgaussian import config

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

from simple import gstrainer

@hydra.main(
    version_base=None,
    config_path=osp.join(_project_dir, "config/"),
    config_name="surf2dgaussian"
)
def main(_cfg: DictConfig):
    OmegaConf.resolve(_cfg)
    #print(OmegaConf.to_yaml(_cfg))
    _config = config.load_surf2dgaussian_config(_cfg)

    _config.scene.out_path = HydraConfig.get().runtime.output_dir
    _config.scene.log_path = HydraConfig.get().runtime.output_dir

    _config.sys.dataCls = "gausiansplatting.data.nerf.NerfSource"
    _config.sys.sceneCls = "surf2dgaussian.scene.surf_scene.Scene"
    _config.sys.modelCls = "surf2dgaussian.scene.surf_model.Module"
    _config.sys.policyCls = "surf2dgaussian.scene.surf_policy.Policy"
    _config.sys.optimerCls = "gausiansplatting.scene.gs_optimer.Optimer"
    _config.sys.lossCls = "surf2dgaussian.scene.surf_loss.Loss"
    _config.sys.renderCls = "surf2dgaussian.scene.surf_render.Render"

    gstrainer.main(_config)


if __name__ == "__main__":
    main()
