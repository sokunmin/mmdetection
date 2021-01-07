import torch
import os.path as osp
from collections import OrderedDict
from mmcv.runner import load_state_dict


def load_checkpoint(model,
                    filename,
                    map_location=None,
                    strict=False,
                    logger=None,
                    show_model_arch=True,
                    print_keys=True):
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
                if 'deconv_layers' in new_k:
                    new_k = new_k.replace("conv_offset_mask.", "conv_offset.")
                    new_k = new_k.replace("bbox_head.deconv_layers.", "neck.upsamples.")
                    if '.0.0.' in new_k:
                        new_k = new_k.replace(".0.0.", ".0.dcn.")
                    if '.0.1.' in new_k:
                        new_k = new_k.replace(".0.1.", ".0.dcn_bn.")
                    if '.1.0.' in new_k:
                        new_k = new_k.replace(".1.0.", ".1.dcn.")
                    if '.1.1.' in new_k:
                        new_k = new_k.replace(".1.1.", ".1.dcn_bn.")
                    if '.2.0.' in new_k:
                        new_k = new_k.replace(".2.0.", ".2.dcn.")
                    if '.2.1.' in new_k:
                        new_k = new_k.replace(".2.1.", ".2.dcn_bn.")

                if '.shortcut_layers.' in new_k:
                    new_k = new_k.replace("bbox_head.shortcut_layers.", "neck.shortcuts.")
                    new_k = new_k.replace(".layers.", ".")
                if '.hm.' in new_k:
                    new_k = new_k.replace(".hm.", ".ct_hm_head.")
                if '.wh.' in new_k:
                    new_k = new_k.replace(".wh.", ".ct_wh_head.")

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