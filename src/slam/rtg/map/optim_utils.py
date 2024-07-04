import torch
from torch.nn import functional as F

def _transmission2tilemask(pixelmask, stride, tile_mask_ratio=0.5):
    height, width = pixelmask.shape[:2]
    pad_h = (height + stride - 1) // stride * stride - height
    pad_w = (width + stride - 1) // stride * stride - width
    pixelmask_pad = F.pad(pixelmask, (0, pad_w, 0, pad_h))
    tilemask = F.avg_pool2d(
        pixelmask_pad.unsqueeze(0).unsqueeze(0).float(),
        kernel_size=stride,
        stride=stride,
    )
    return (tilemask > tile_mask_ratio).int().squeeze(0).squeeze(0)

def _colorerror2tilemask(color_error, stride, top_ratio=0.4):
    height, width = color_error.shape[:2]
    pad_h = (height + stride - 1) // stride * stride - height
    pad_w = (width + stride - 1) // stride * stride - width
    color_error_pad = F.pad(color_error, (0, pad_w, 0, pad_h), value=0)

    color_error_downscale = (
        F.avg_pool2d(
            color_error_pad.unsqueeze(0).unsqueeze(0).float(),
            kernel_size=stride,
            stride=stride,
        )
        .squeeze(0)
        .squeeze(0)
    )
    sample_num = int(torch.numel(color_error_downscale) * top_ratio)
    _, color_error_index = torch.topk(color_error_downscale.view(-1), k=sample_num)
    top_values_indices = torch.stack(
        [
            color_error_index // color_error_downscale.shape[1],
            color_error_index % color_error_downscale.shape[1],
        ],
        dim=1,
    )
    tile_mask = torch.zeros_like(color_error_downscale, dtype= torch.int32, device= color_error.device)
    tile_mask[top_values_indices[:, 0], top_values_indices[:, 1]] = 1
    return tile_mask


def evaluate_render_range(_output, _frameData, global_opt = False, sample_ratio = -1):

    _T_map = _output["transmission"]

    if global_opt:
        if sample_ratio > 0:
            _rgb = _output["color"]
            _gt_rgb = _frameData.params["color"]
            _diff = (_rgb - _gt_rgb).abs()
            _c_error = torch.sum(_diff, dim=-1, keepdim=False)

            filter_mask = (_rgb.sum(dim=-1) == 0)
            _c_error[filter_mask] = 0

            tile_mask = _colorerror2tilemask(_c_error, 16, sample_ratio)
            render_mask = (
                F.interpolate(
                    tile_mask.float().unsqueeze(0).unsqueeze(0),
                    scale_factor=16,
                    mode="nearest",
                )
                .squeeze(0)
                .squeeze(0)
                .bool()
            )[: _c_error.shape[0], : _c_error.shape[1]]
        # after training, real global optimization
        else:
            render_mask = (_T_map != 1).squeeze(-1)
            tile_mask = None
    else:
        render_mask = (_T_map != 1).squeeze(-1)
        tile_mask = _transmission2tilemask(render_mask, 16, 0.5)

    return render_mask, tile_mask
