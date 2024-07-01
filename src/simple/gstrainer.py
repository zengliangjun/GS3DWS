import os
import torch
import hydra
from simple import logger
from tqdm import tqdm
from simple import gsviewer

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
    _policyCls = hydra.utils.get_class(_sysConfig.policyCls)
    _optimerCls = hydra.utils.get_class(_sysConfig.optimerCls)
    _lossCls = hydra.utils.get_class(_sysConfig.lossCls)
    _renderCls = hydra.utils.get_class(_sysConfig.renderCls)

    _gsscene = _sceneCls(_config)
    _gsmodel = _modelCls(_config)
    _gspolicy = _policyCls(_config)
    _gsoptimer = _optimerCls(_config)
    _gsloss = _lossCls(_config)
    _gsrender = _renderCls(_config)

    _device = _gsscene.device()

    _startiter = 0
    _paramters = _gsscene.load()
    if _paramters is not None:
        _gsmodel.merger(_paramters)
        if hasattr(_paramters, 'iter'):
            _startiter = _paramters['iter']

    else:
        _rawPoints = _gsscene.loadRawPoints()
        _rawPoints = _rawPoints.to(_device)
        _paramters = _gsmodel.build(_rawPoints)
        _gsmodel.merger(_paramters)

    _gsmodel.train()

    _gsoptimer.setup(_gsmodel.params())

    _emaloss4log = 0
    _optimConfig = _config.optim
    _bar = tqdm(range(_startiter, _optimConfig.iterations), desc="Training progress")
    _startiter += 1
    for _iter in range(_startiter, _optimConfig.iterations + 1):

        _frame = next(_gsscene)
        _rout = _gsrender(_frame, _gsmodel.renderparams())

        _loss, _losses = _gsloss(_frame, _rout)
        _loss.backward()

        for _key in _losses:
            _losses[_key] = _losses[_key].item()

        _losses['num'] = _gsmodel.num

        _writer.write_dict(_losses, _iter)

        with torch.no_grad():
            _gspolicy.update(_iter, _gsmodel, _gsoptimer, _rout)

            _gsoptimer.step(_iter)

            _gsscene.save(_iter, _gsmodel)

            _emaloss4log = 0.4 * _loss.item() + 0.6 * _emaloss4log

            if _iter % 10 == 0:
                _bar.set_postfix({"Loss": f"{_loss.item():.{7}f} / {_emaloss4log:.{7}f}"})
                _bar.update(10)
            if _iter == _optimConfig.iterations:
                _bar.close()

            gsviewer._stepviewgui(_config, _gsrender, _gsmodel)

    _gsscene.save(_iter, _gsmodel)
