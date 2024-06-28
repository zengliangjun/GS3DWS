from dataclasses import dataclass, field

@dataclass
class DataCfg:
    name: str  # [tum  ]
    eval: bool #
    source_path: str

@dataclass
class SceneCfg:
    model_path: str
    scene_color: list[int]# = field(default_factory=list) # [0, 0, 0]

    saving_iterations: list[int]

    sh_degree: int# = 3
    device: str# = "cuda:0"

    data: DataCfg

    out_path: str
    log_path: str


@dataclass
class RenderCfg:
    convert_SHs_python: bool# = False
    compute_cov3D_python: bool# = False
    debug: bool# = False
    sh_degree: int
    bg_color: list[int]

@dataclass
class OptimCfg:
    iterations: int# = 30_000
    saving_iterations: list[int]

    lambda_dssim: float# = 0.2

    position_lr_init: float#  = 0.00016
    position_lr_final: float# = 0.0000016
    position_lr_delay_mult: float# = 0.01
    position_lr_max_steps: int# = 30_000
    
    feature_lr: float# = 0.0025
    opacity_lr: float# = 0.05
    scaling_lr: float# = 0.005
    rotation_lr: float# = 0.001

@dataclass
class PolicyCfg:
    percent_dense: float# = 0.01

    densification_interval: int# = 100
    opacity_reset_interval: int# = 3000

    densify_from_iter: int# = 500
    densify_until_iter: int# = 15_000

    densify_grad_threshold: float# = 0.0002
    random_background: bool# = False

@dataclass
class SysConfig:
    dataCls: str
    sceneCls: str
    modelCls: str
    policyCls: str
    optimerCls: str
    lossCls: str
    renderCls: str


@dataclass
class GSTrainConfig:
    scene: SceneCfg
    render: RenderCfg
    optim: OptimCfg
    policy: PolicyCfg
    sys: SysConfig


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

def load_gausiansplatting_config(cfg: DictConfig) -> GSTrainConfig:
    return load_typed_config(cfg, GSTrainConfig, {})
