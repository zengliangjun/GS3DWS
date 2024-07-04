from dataclasses import dataclass


from omegaconf import DictConfig
from gausiansplatting import config as gsconfig


@dataclass
class RenderCfg:
    bg_color: list[int]
    active_sh_degree: int
    renderer_opaque_threshold: float = 0.6
    renderer_normal_threshold: int = 60
    renderer_depth_threshold: float = 1.0
    color_sigma: float = 3.0

@dataclass
class PreprocessCfg:
    min_depth: float = 0.3
    max_depth: float = 5.0
    depth_filter: bool = False
    invalid_confidence_thresh: float = 0.2

@dataclass
class GaussianCfg:
    xyz_factor: list #= [1, 1, 0.1] # z should be smallest
    init_opacity: float = 0.99
    scale_factor: float = 1.0
    max_radius: float = 0.05
    min_radius: float = 0.001

@dataclass
class StateCfg:
    # state manage
    memory_length: int = 5
    keyframe_trans_thes: float = 0.3
    keyframe_theta_thes: float = 30


@dataclass
class MapCfg:
    preprocess: PreprocessCfg
    gaussian: GaussianCfg
    state: StateCfg

    uniform_sample_num: int = 50000
    transmission_sample_ratio: float = 1.0
    error_sample_ratio: float = 0.05

    add_transmission_thres: float = 0.5
    add_depth_thres: float = 0.1
    add_color_thres: float = 0.1
    add_normal_thres: float = 1000
    history_merge_max_weight: float = 0.5

    stable_confidence_thres: float = 500
    unstable_time_window: float = 200
    KNN_num: int = 15
    KNN_threshold: int = -1

@dataclass
class OptimCfg:
    # optimize params:
    gaussian_update_iter: int = 100
    final_global_iter: int = 10

    gaussian_update_frame: int = 5
    global_keyframe_num: int = 3

    color_weight: float = 0.8
    depth_weight: float = 1.0
    ssim_weight: float = 0.2
    normal_weight: float = 0.0
    position_lr : float = 0.001
    feature_lr : float = 0.0005
    opacity_lr : float = 0.000
    scaling_lr : float = 0.004
    rotation_lr : float = 0.001
    feature_lr_coef: float = 1.0
    scaling_lr_coef: float = 1.0
    rotation_lr_coef: float = 1.0

@dataclass
class SysCfg:
    dataCls: str
    frameCls: str
    trackCls: str
    mapCls: str
    modelCls: str
    optimizerCls: str
    renderCls: str
    stateCls: str

@dataclass
class GSTrainCfg:
    scene: gsconfig.SceneCfg
    render: RenderCfg
    mapper: MapCfg
    optim: OptimCfg
    sys: SysCfg

def load_rtgslam_config(cfg: DictConfig) -> GSTrainCfg:
    return gsconfig.load_typed_config(cfg, GSTrainCfg, {})
