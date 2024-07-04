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

from slam.rtg import config, rtgslam

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=osp.join(_project_dir, "config/"),
    config_name="rtg_slam_tum_freiburg3_office_household"
)
def main(_cfg: DictConfig):
    OmegaConf.resolve(_cfg)
    #print(OmegaConf.to_yaml(_cfg))
    _config = config.load_rtgslam_config(_cfg)

    _config.sys.dataCls = "slam.data.mtu.rgbd.RGBDSource"
    _config.sys.frameCls = "slam.rtg.frame.Frame"
    _config.sys.trackCls = "slam.rtg.notracker.Tracker"
    _config.sys.mapCls = "slam.rtg.mapper.Mapper"

    _config.sys.modelCls = "slam.rtg.map.model.SLAMModel"
    _config.sys.optimizerCls = "slam.rtg.map.optimizer.Optimizer"
    _config.sys.renderCls = "slam.rtg.map.render.Render"
    _config.sys.stateCls = "slam.rtg.map.state.Manager"


    rtgslam.main(_config)


if __name__ == "__main__":
    main()
