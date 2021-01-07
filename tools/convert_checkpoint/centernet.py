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
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
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
            if 'stage0.' in new_k:
                if "stage0.0.":
                    new_k = new_k.replace("stage0.0.", "conv1.")
                if "stage0.1.":
                    new_k = new_k.replace("stage0.1.", "bn1.")
            if 'stage1.' in new_k:
                new_k = new_k.replace("stage1.", "layer1.")
            if 'stage2.' in new_k:
                new_k = new_k.replace("stage2.", "layer2.")
            if 'stage3.' in new_k:
                new_k = new_k.replace("stage3.", "layer3.")
            if 'stage4.' in new_k:
                new_k = new_k.replace("stage4.", "layer4.")
            if 'upsample.deconv' in new_k:
                new_k = new_k.replace("upsample.", "neck.upsamples.")
                if 'deconv1' in new_k:
                    new_k = new_k.replace("deconv1.", "0.")
                if 'deconv2' in new_k:
                    new_k = new_k.replace("deconv2.", "1.")
                if 'deconv3' in new_k:
                    new_k = new_k.replace("deconv3.", "2.")
                if 'offset_mask_conv.' in new_k:
                    new_k = new_k.replace("offset_mask_conv.", "conv_offset.")
                if 'dcnv2.' in new_k:
                    new_k = new_k.replace("dcnv2.", "")
                if '.up_sample.' in new_k:
                    new_k = new_k.replace(".up_sample.", ".upsample.")
                elif '.up_bn.' in new_k:
                    new_k = new_k.replace(".up_bn.", ".upsample_bn.")
            if 'head.' in new_k:
                if 'head.cls_head.feat_conv' in new_k:
                    new_k = new_k.replace("head.cls_head.feat_conv", "bbox_head.ct_hm_head.0.conv")
                if 'head.cls_head.out_conv' in new_k:
                    new_k = new_k.replace("head.cls_head.out_conv", "bbox_head.ct_hm_head.1")
                if 'head.wh_head.feat_conv' in new_k:
                    new_k = new_k.replace("head.wh_head.feat_conv", "bbox_head.ct_wh_head.0.conv")
                if 'head.wh_head.out_conv' in new_k:
                    new_k = new_k.replace("head.wh_head.out_conv", "bbox_head.ct_wh_head.1")
                if 'head.reg_head.feat_conv' in new_k:
                    new_k = new_k.replace("head.reg_head.feat_conv", "bbox_head.ct_reg_head.0.conv")
                if 'head.reg_head.out_conv' in new_k:
                    new_k = new_k.replace("head.reg_head.out_conv", "bbox_head.ct_reg_head.1")

            if print_keys:
                print('> key = ', k, ' -> ', new_k)
            new_state_dict[new_k] = v
    # load state_dict
    if hasattr(model, 'module'):
        load_state_dict(model.module, new_state_dict, strict, logger)
    else:
        load_state_dict(model, new_state_dict, strict, logger)
    return checkpoint