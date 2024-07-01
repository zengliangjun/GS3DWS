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


from dacite import Config, from_dict
from omegaconf import DictConfig, OmegaConf
from typing import Type, TypeVar
from pathlib import Path

TYPE_HOOKS = {
    Path: Path,
}

T = TypeVar("T")

def load_typed_config(
    cfg: DictConfig,
    data_class: Type[T],
    extra_type_hooks: dict = {},
) -> T:
    return from_dict(data_class, OmegaConf.to_container(cfg),
        config=Config(type_hooks={**TYPE_HOOKS, **extra_type_hooks}))

def load_surf2dgaussian_config(cfg: DictConfig) -> SurfTrainConfig:
    return load_typed_config(cfg, SurfTrainConfig, {})
