import torch
import os.path as osp
from collections import OrderedDict
from mmcv.runner import load_state_dict


def load_checkpoint(model,
                    filename,
                    map_location=None,
                    strict=False,
                    logger=None,
                    show_model_arch=False,
                    print_keys=False):
    """ Note that official pre-trained models use `GroupNorm` in backbone.
      """
    if not osp.isfile(filename):
        raise IOError('{} is not a checkpoint file'.format(filename))
    checkpoint = torch.load(filename, map_location=map_location)
    # get state_dict from checkpoint
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise RuntimeError(
            'No state_dict found in checkpoint file {}'.format(filename))
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith('backbone.'):
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            new_k = k
            if 'bbox_head.' in new_k:
                new_k = new_k.replace("conv_offset_mask.", "conv_offset.")

            if print_keys:
                print('> key = ', k, ' -> ', new_k)
            state_dict[new_k] = v
    if show_model_arch:
        print('> model = ', model)
    # load state_dict
    if hasattr(model, 'module'):
        load_state_dict(model.module, state_dict, strict, logger)
    else:
        load_state_dict(model, state_dict, strict, logger)
    return checkpoint