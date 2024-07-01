import os
import torch
from torch.nn import functional as F
import hydra
from simple import logger
from common.view import network_gui
from gausiansplatting.scene import gs_scene
from surf2dgaussian.scene import surf_render
import matplotlib.pyplot as plt
import traceback

def gradient_map(image):
    _dtype = image.dtype
    _device = image.device

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype= _dtype, device= _device).unsqueeze(0).unsqueeze(0)/4
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype= _dtype, device= _device).unsqueeze(0).unsqueeze(0)/4
    
    grad_x = torch.cat([F.conv2d(image[i].unsqueeze(0), sobel_x, padding=1) for i in range(image.shape[0])])
    grad_y = torch.cat([F.conv2d(image[i].unsqueeze(0), sobel_y, padding=1) for i in range(image.shape[0])])
    magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    magnitude = magnitude.norm(dim=0, keepdim=True)
    return magnitude

def colormap(map, cmap="turbo"):
    colors = torch.tensor(plt.cm.get_cmap(cmap).colors).to(map.device)
    map = (map - map.min()) / (map.max() - map.min())
    map = (map * 255).round().long().squeeze()
    map = colors[map].permute(2,0,1)
    return map


def _preset_torch():
    torch.autograd.set_detect_anomaly(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.fastest = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True


def _prepare_output(args):
    os.makedirs(args.out_path, exist_ok = True)
    os.makedirs(args.log_path, exist_ok = True)
    logger.configure(args.log_path)
    logger.log(f"Output folder: {args.out_path}")

def render_net_image(render_pkg, render_items, render_mode, camera):
    output = render_items[render_mode].lower()
    if output == 'alpha':
        net_image = render_pkg["alpha"]
        net_image = (net_image - net_image.min()) / (net_image.max() - net_image.min())
    elif output == 'depth':
        net_image = render_pkg["median"]
        net_image = (net_image - net_image.min()) / (net_image.max() - net_image.min())
    elif output == 'normal':
        net_image = surf_render.depth_to_normal(camera, render_pkg["median"]).permute(2,0,1)
        net_image = (net_image+1)/2
    elif output == 'edge':
        net_image = gradient_map(render_pkg["color"])
    elif output == 'curvature':
        net_image = gradient_map(surf_render.depth_to_normal(camera, render_pkg["median"]).permute(2,0,1))
    else:
        net_image = render_pkg["color"]
    
    #Make sure the rendering image is Shape of (3, H, W) or (1, H, W)
    if net_image.shape[0] == 1:
        net_image = colormap(net_image)
    return net_image 

def _initviewgui(_config):
    _ip = "127.0.0.1"
    if hasattr(_config, "bindip"):
        _ip = _config.bindip
    _port = 6009
    if hasattr(_config, "port"):
        _port = _config.port

    network_gui.init(_ip, _port)
    _render_items = ['RGB', 'Alpha', 'Depth', 'Normal', 'Curvature', 'Edge']
    if not hasattr(_config, "render_items"):
        setattr(_config, "render_items", _render_items)

def _stepviewgui(_config, _gsrender, _gsmodel):
    if network_gui.conn is None:
        network_gui.try_connect(_config.render_items)

    if network_gui.conn is None:
        return

    try:
        _bytes = None
        _items = network_gui.receive()
        if _items is not None:

            _org_scaling_modifier = _gsrender.scaling_modifier
            _gsrender.scaling_modifier = float(_items['scaling_modifier'])

            _rout = _gsrender(_items['frame'], _gsmodel.renderparams())
            _gsrender.scaling_modifier = _org_scaling_modifier

            _image = render_net_image(_rout, _config.render_items, _items['mode'], _items['frame'])
            _bytes = memoryview((torch.clamp(_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())

        metrics_dict = {
            "#": _gsmodel.points_num
            # Add more metrics as needed
        }
        import os.path as osp
        _source_path = osp.abspath(_config.scene.data.source_path)
        network_gui.send(_bytes, _source_path, metrics_dict)

    except:
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        traceback.print_exc()
        del network_gui.conn
        network_gui.conn = None

def main(_config):  ##  GSTrainConfig
    _preset_torch()
    _prepare_output(_config.scene)

    _sysConfig = _config.sys
    _device = torch.device(_config.scene.device)
    _modelCls = hydra.utils.get_class(_sysConfig.modelCls)
    _renderCls = hydra.utils.get_class(_sysConfig.renderCls)

    _gsmodel = _modelCls(_config)
    _gsrender = _renderCls(_config)

    _paramters = gs_scene.load_toolkit(_config.scene, _device)
    if _paramters is not None:
        _gsmodel.merger(_paramters)
    else:
        raise f"load error from {_config.scene.model_path}"

    _gsmodel.eval()

    for _i in range(_config.scene.sh_degree):
        _gsmodel.oneupSHdegree()

    _initviewgui(_config)

    while True:
        _stepviewgui(_config, _gsrender, _gsmodel)
