from gausiansplatting import config as gsconfig
from dataclasses import dataclass


class OptimCfg(gsconfig.OptimCfg):
    lambda_dist : float = 0.0
    lambda_normal : float = 0.05
    opacity_cull : float = 0.05


@dataclass
class RenderCfg(gsconfig.RenderCfg):
    depth_ratio : int = 0  ## 0 for mean depth and 1 for median depth

@dataclass
class SurfTrainConfig:
    scene: gsconfig.SceneCfg
    render: RenderCfg
    optim: OptimCfg
    policy: gsconfig.PolicyCfg
    sys: gsconfig.SysConfig

from omegaconf import DictConfig

def load_surf2dgaussian_config(cfg: DictConfig) -> SurfTrainConfig:
    return gsconfig.load_typed_config(cfg, SurfTrainConfig, {})
