import os.path as osp
import sys
import hydra
from omegaconf import DictConfig, OmegaConf

_project_dir = osp.join(osp.dirname(__file__), "../../")
_project_dir = osp.abspath(_project_dir)

def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)

# src python path
add_pypath(osp.join(_project_dir, "src"))

from gausiansplatting import config

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

from simple import gsrender

@hydra.main(
    version_base=None,
    config_path=osp.join(_project_dir, "config/"),
    config_name="gaussian_splatting"
)
def main(_cfg: DictConfig):
    OmegaConf.resolve(_cfg)
    #print(OmegaConf.to_yaml(_cfg))
    _config = config.load_gausiansplatting_config(_cfg)

    _config.scene.data.eval = True
    _config.scene.out_path = "/tmp"
    _config.scene.log_path = "/tmp/log"
    _config.scene.model_path = "outputs/Gaussian_Splatting/lego/20240701_175938"

    _config.sys.dataCls = "gausiansplatting.data.nerf.NerfSource"
    _config.sys.sceneCls = "gausiansplatting.scene.gs_scene.Scene"
    _config.sys.modelCls = "gausiansplatting.scene.gs_model.Module"
    _config.sys.lossCls = "gausiansplatting.scene.gs_loss.Loss"
    #_config.sys.renderCls = "gausiansplatting.scene.gs_render.Render"
    _config.sys.renderCls = "extends.gausianmonitor.scene.monitor_render.Render"

    gsrender.main(_config)

if __name__ == "__main__":
    main()
