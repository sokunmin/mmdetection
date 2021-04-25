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
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise RuntimeError(
            'No state_dict found in checkpoint file {}'.format(filename))

    if show_model_arch:
        print('> model = ', model)

    # strip prefix of state_dict
    new_state_dict = {}
    if list(state_dict.keys())[0].startswith('base'):
        for k, v in state_dict.items():
            new_k = k
            if 'base.' in new_k:
                new_k = new_k.replace("base.", "backbone.base_layers.")

            if 'last_proj.' in new_k:
                new_k = new_k.replace("last_proj.", "neck.last_proj.")

            if 'last_blk.' in new_k:
                new_k = new_k.replace("last_blk.", "neck.last_block.")

            if 'denseBlocksUp.' in new_k:
                new_k = new_k.replace("denseBlocksUp.", "neck.dense_up_blocks.")

            if 'conv1x1_up.' in new_k:
                new_k = new_k.replace("conv1x1_up.", "neck.conv1x1_up.")

            if 'hm.' in new_k:
                new_k = new_k.replace("hm.", "bbox_head.ct_hm_head.")
                new_k = new_k.replace(".0.", ".0.conv.")
                new_k = new_k.replace("bbox_head.ct_hm_head.2.", "bbox_head.ct_hm_head.1.")

            if 'wh.' in new_k:
                new_k = new_k.replace("wh.", "bbox_head.ct_wh_head.")
                new_k = new_k.replace(".0.", ".0.conv.")
                new_k = new_k.replace("bbox_head.ct_wh_head.2.", "bbox_head.ct_wh_head.1.")

            if '.norm.' in new_k:
                new_k = new_k.replace(".norm.", ".bn.")

            if print_keys:
                print('> key = ', k, ' -> ', new_k)
            new_state_dict[new_k] = v
    # load state_dict
    if hasattr(model, 'module'):
        load_state_dict(model.module, new_state_dict, strict, logger)
    else:
        load_state_dict(model, new_state_dict, strict, logger)
    return checkpoint