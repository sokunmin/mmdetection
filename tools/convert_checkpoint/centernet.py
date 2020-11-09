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
                    print_keys=True):
    """ Note that official pre-trained models use `GroupNorm` in backbone.
      """
    if not osp.isfile(filename):
        raise IOError('{} is not a checkpoint file'.format(filename))
    checkpoint = torch.load(filename, map_location=map_location)
    # get state_dict from checkpoint
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise RuntimeError(
            'No state_dict found in checkpoint file {}'.format(filename))

    if show_model_arch:
        print('> model = ', model)

    # strip prefix of state_dict
    new_state_dict = {}
    if list(state_dict.keys())[0].startswith('backbone'):
        for k, v in state_dict.items():
            new_k = k
            if 'backbone_model.' in new_k:
                new_k = new_k.replace("backbone_model.", "backbone.")
            if 'conv_offset_mask.' in new_k:
                new_k = new_k.replace("conv_offset_mask.", "conv_offset.")
            if 'head_model.' in new_k:
                new_k = new_k.replace("head_model.", "bbox_head.")
                if '0.weight' in new_k:
                    new_k = new_k.replace("0.weight", "0.conv.weight")
                if '0.bias' in new_k:
                    new_k = new_k.replace("0.bias", "0.conv.bias")
                if '2.weight' in new_k:
                    new_k = new_k.replace("2.weight", "1.weight")
                if '2.bias' in new_k:
                    new_k = new_k.replace("2.bias", "1.bias")

            if print_keys:
                print('> key = ', k, ' -> ', new_k)
            new_state_dict[new_k] = v
    # load state_dict
    if hasattr(model, 'module'):
        load_state_dict(model.module, new_state_dict, strict, logger)
    else:
        load_state_dict(model, new_state_dict, strict, logger)
    return checkpoint