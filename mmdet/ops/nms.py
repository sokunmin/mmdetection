import torch.nn as nn


def simple_nms(heat, kernel=3, out_heat=None):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(heat,
                                    (kernel, kernel),
                                    stride=1,
                                    padding=pad)
    keep = (hmax == heat).float()
    out_heat = heat if out_heat is None else out_heat
    return out_heat * keep