from simple import gstrainer
from simple import logger
import os.path as osp
import hydra
#from simple import gsviewer

def main(_config):  ##  GSTrainConfig
    #setattr(_config, "render_items", ['RGB'])
    #gsviewer._initviewgui(_config)

    gstrainer._preset_torch()
    gstrainer._prepare_output(_config.scene)

    _writer = logger.Visualizer(osp.join(logger.get_dir(), 'tf_events'))

    _sysConfig = _config.sys
    _dataCls = hydra.utils.get_class(_sysConfig.dataCls)
    _frameCls = hydra.utils.get_class(_sysConfig.frameCls)
    _trackCls = hydra.utils.get_class(_sysConfig.trackCls)
    _mapCls = hydra.utils.get_class(_sysConfig.mapCls)

    _modelCls = hydra.utils.get_class(_sysConfig.modelCls)
    _optimizerCls = hydra.utils.get_class(_sysConfig.optimizerCls)
    _renderCls = hydra.utils.get_class(_sysConfig.renderCls)
    _stateCls = hydra.utils.get_class(_sysConfig.stateCls)


    _data = _dataCls(_config)
    #_frame = _frameCls(_config)
    _tracker = _trackCls(_config)
    _map = _mapCls(_config)
    _model = _modelCls(_config)
    _optimizer = _optimizerCls(_config)
    _render = _renderCls(_config)
    _state = _stateCls(_config)

    #setattr(_config, "slam_data", _data)
    #setattr(_config, "slam_frame", _frame)
    #setattr(_config, "slam_track", _track)
    setattr(_config, "slam_map", _map)
    setattr(_config, "slam_model", _model)
    setattr(_config, "slam_optimizer", _optimizer)
    setattr(_config, "slam_render", _render)
    setattr(_config, "slam_state", _state)
    setattr(_config, "writer", _writer)

    _dataIte = iter(_data)
    while True:

        _frameData = next(_dataIte)
        if _frameData is None:
            break
        _frameData = _frameCls(_frameData)
        _poses = _tracker.tracking(_frameData)
        _map.update(_poses, _frameData)

        #gsviewer._stepviewgui(_config, _render, _model)

    _map.final()
