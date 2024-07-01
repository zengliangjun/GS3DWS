import os
import os.path as osp
import torch
import hydra
from simple import logger
from tqdm import tqdm
from simple import gsviewer
import torchvision

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

    # TODO
    #with open(os.path.join(args.scene.out_path, "cfg_args"), 'w') as cfg_log_f:
    #    cfg_log_f.write(str(Namespace(**vars(args))))

def main(_config):  ##  GSTrainConfig
    gsviewer._initviewgui(_config)

    _preset_torch()
    _prepare_output(_config.scene)

    _writer = logger.Visualizer(os.path.join(logger.get_dir(), 'tf_events'))

    _sysConfig = _config.sys
    _sceneCls = hydra.utils.get_class(_sysConfig.sceneCls)
    _modelCls = hydra.utils.get_class(_sysConfig.modelCls)
    _lossCls = hydra.utils.get_class(_sysConfig.lossCls)
    _renderCls = hydra.utils.get_class(_sysConfig.renderCls)

    _gsscene = _sceneCls(_config)
    _gsmodel = _modelCls(_config)
    _gsloss = _lossCls(_config)
    _gsrender = _renderCls(_config)

    _paramters = _gsscene.load()
    if _paramters is not None:
        _gsmodel.merger(_paramters)
    else:
        raise Exception("")

    _gsmodel.eval()

    _emaloss4log = 0

    _bar = tqdm(range(0, len(_gsscene)), desc="Training progress")
    _outdir = osp.join(_config.scene.model_path, "render")
    if not osp.exists(_outdir):
        os.makedirs(_outdir)

    for _iter in range(0, len(_gsscene)):

        _frame = next(_gsscene)
        _rout = _gsrender(_frame, _gsmodel.renderparams())
        _color = _rout["color"]
        _file_name = f'{_frame.color_name}.png'
        _file_name = osp.basename(_file_name)
        torchvision.utils.save_image(_color, os.path.join(_outdir, _file_name))

        _loss, _losses = _gsloss(_frame, _rout)

        #for _key in _losses:
        #    _losses[_key] = _losses[_key].item()
        #_writer.write_dict(_losses, _iter)

        _emaloss4log = 0.4 * _loss.item() + 0.6 * _emaloss4log

        if _iter % 10 == 0:
            _bar.set_postfix({"Loss": f"{_loss.item():.{7}f} / {_emaloss4log:.{7}f}"})
            _bar.update(10)
        if _iter == len(_gsscene):
            _bar.close()

        gsviewer._stepviewgui(_config, _gsrender, _gsmodel)
