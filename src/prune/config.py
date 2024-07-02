from gausiansplatting import config as gsconfig
from dataclasses import dataclass
from omegaconf import DictConfig

### compact_3dgs
@dataclass
class PolicyCompact3dgs(gsconfig.PolicyCfg):
    mask_prune_iter : int  = 1_000

@dataclass
class OptimCompact3dgs(gsconfig.OptimCfg):
    mask_lr : float = 0.01
    lambda_mask : float = 0.0005

@dataclass
class Compact3dgsConfig:
    scene: gsconfig.SceneCfg
    render: gsconfig.RenderCfg
    optim: OptimCompact3dgs
    policy: PolicyCompact3dgs
    sys: gsconfig.SysConfig

def load_compact3dgs_config(cfg: DictConfig) -> Compact3dgsConfig:
    return gsconfig.load_typed_config(cfg, Compact3dgsConfig, {})


### mini splatting ms
@dataclass
class SceneMinims(gsconfig.SceneCfg):
    imp_metric : str # ['outdoor'  'indoor'] 

@dataclass
class PolicyMinims(gsconfig.PolicyCfg):
    simp_iteration1 : int  = 15_000
    simp_iteration2 : int  = 20_000
    num_depth: int  = 3_500_000
    num_max: int  = 4_500_000
    sampling_factor: float = 0.5

@dataclass
class MinimsConfig:
    scene: SceneMinims
    render: gsconfig.RenderCfg
    optim: gsconfig.OptimCfg
    policy: PolicyMinims
    sys: gsconfig.SysConfig

def load_minims_config(cfg: DictConfig) -> MinimsConfig:
    _config = gsconfig.load_typed_config(cfg, MinimsConfig, {})
    ## ###
    _config.scene.sh_degree = 0
    _config.scene.sh_degree = 0
