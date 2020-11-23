import torch.nn as nn

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from mmdet.models import build_loss
from ...core.bbox.transforms import keypoint2result


@DETECTORS.register_module()
class SingleStageMultiDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 mask_head=None,
                 keypoint_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 loss_auto_weighted=None):
        super(SingleStageMultiDetector, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        if bbox_head is not None:
            self.bbox_head = build_head(bbox_head)
        if mask_head is not None:
            self.mask_head = build_head(mask_head)
        if keypoint_head is not None:
            self.keypoint_head = build_head(keypoint_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if loss_auto_weighted is not None:
            self.loss_auto_weighted = build_loss(loss_auto_weighted)
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(SingleStageMultiDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_bbox:
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
        if self.with_keypoint:
            self.keypoint_head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def group_inputs(self, x, outs=None, key='bbox'):
        return x, outs

    def group_outputs(self, *inputs, key='bbox', return_loss=False):
        return inputs

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        group_keys = []
        x = self.extract_feat(img)
        all_outs = {}
        if self.with_bbox:
            bbox_outs = self.bbox_head(x)
            all_outs[group_keys[-1]] = bbox_outs
        if self.with_mask:
            x, all_outs = self.group_inputs(x, all_outs, key='mask')
            mask_outs = self.mask_head(x)
            all_outs['mask'] = mask_outs
        if self.with_keypoint:
            x, all_outs = self.group_inputs(x, all_outs, key='keypoint')
            keypoint_outs = self.keypoint_head(x)
            all_outs['keypoint'] = keypoint_outs
        return all_outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_masks_ignore=None,
                      gt_keypoints=None,
                      gt_keypoints_ignore=None):
        x = self.extract_feat(img)
        group_keys = []
        if self.with_shared_head:
            x = self.shared_head(x)
            x = self.group_inputs(x, key='fpn')
        all_outs = {}
        if self.with_bbox:
            x, all_outs = self.group_inputs(x, all_outs, key='bbox')
            bbox_outs = self.bbox_head(x)
            all_outs['bbox'] = bbox_outs
        if self.with_mask:
            x, all_outs = self.group_inputs(x, all_outs, key='mask')
            mask_outs = self.mask_head(x)
            all_outs['mask'] = mask_outs
        if self.with_keypoint:
            x, all_outs = self.group_inputs(x, all_outs, key='keypoint')
            keypoint_outs = self.keypoint_head(x)
            all_outs['keypoint'] = keypoint_outs
        gt_inputs = (gt_bboxes, gt_masks, gt_keypoints, gt_labels)
        losses = self.loss(all_outs, gt_inputs, img_metas,
                           gt_bboxes_ignore=gt_bboxes_ignore,
                           gt_masks_ignore=gt_masks_ignore,
                           gt_keypoints_ignore=gt_keypoints_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        x = self.extract_feat(img)
        if self.with_shared_head:
            x = self.shared_head(x)
            x = self.group_inputs(x, key='fpn')

        all_outs = {}
        all_results = [None, None, None]
        if self.with_bbox:
            all_outs['bbox'] = self.bbox_head(x)
            bbox_inputs = self.group_outputs(all_outs['bbox'], img_metas, self.test_cfg, rescale,
                                             key='bbox')
            # > (n, [bboxes, labels, ...])
            bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
            bbox_results = [
                # convert detection results to a list of numpy arrays.
                bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                for det_bboxes, det_labels in bbox_list
            ]
            all_results[0] = bbox_results[0]
        if self.with_mask:
            x, all_outs = self.group_inputs(x, all_outs, key='mask')
            all_outs['mask'] = self.mask_head(x)
            mask_inputs = self.group_outputs(all_outs['mask'], img_metas, self.test_cfg, rescale,
                                             key='mask')
            mask_list = self.mask_head.get_masks(*mask_inputs)
            all_results[1] = None
            # MY-TODO: add `mask2result`
        if self.with_keypoint:
            x, all_outs = self.group_inputs(x, all_outs, key='keypoint')
            all_outs['keypoint'] = self.keypoint_head(x)
            keypoint_inputs = self.group_outputs(all_outs['keypoint'], img_metas, self.test_cfg, rescale,
                                                 key='keypoint')
            keypoint_list = self.keypoint_head.get_keypoints(*keypoint_inputs)
            keypoint_results = [
                keypoint2result(keypoints, self.MAP_ORDERS, self.keypoint_head.num_joints)
                for keypoints in keypoint_list
            ]
            all_results[2] = keypoint_results[0]
        return all_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

           Args:
               imgs (list[Tensor]): the outer list indicates test-time
                   augmentations and inner Tensor should have a shape NxCxHxW,
                   which contains all images in the batch.
               img_metas (list[list[dict]]): the outer list indicates test-time
                   augs (multiscale, flip, etc.) and the inner list indicates
                   images in a batch. each dict has image information.
               rescale (bool, optional): Whether to rescale the results.
                   Defaults to False.

           Returns:
               list[list[np.ndarray]]: BBox results of each image and classes.
                   The outer list corresponds to each image. The inner list
                   corresponds to each class.
           """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        return [self.bbox_head.aug_test(feats, img_metas, rescale=rescale)]

    def loss(self,
             pred_outs,
             gt_inputs,
             img_metas,
             gt_bboxes_ignore=None,
             gt_masks_ignore=None,
             gt_keypoints_ignore=None):
        gt_bboxes, gt_masks, gt_keypoints, gt_labels = gt_inputs
        all_losses = {}
        if self.with_bbox:
            loss_inputs = self.group_outputs(pred_outs['bbox'], gt_bboxes, gt_labels, img_metas, self.train_cfg,
                                             key='bbox', return_loss=True)
            bbox_losses = self.bbox_head.loss(
                *loss_inputs,
                gt_bboxes_ignore=gt_bboxes_ignore,
                gt_masks_ignore=gt_masks_ignore,
                gt_keypoints_ignore=gt_keypoints_ignore)
            all_losses.update(bbox_losses)
        if self.with_mask:
            loss_inputs = self.group_outputs(pred_outs['mask'], gt_masks, gt_labels, img_metas, self.train_cfg,
                                             key='mask', return_loss=True)
            mask_losses = self.mask_head.loss(
                *loss_inputs,
                gt_bboxes_ignore=gt_bboxes_ignore,
                gt_masks_ignore=gt_masks_ignore,
                gt_keypoints_ignore=gt_keypoints_ignore)
            all_losses.update(mask_losses)
        if self.with_keypoint:
            loss_inputs = self.group_outputs(pred_outs['keypoint'], gt_keypoints, gt_labels, img_metas, self.train_cfg,
                                             key='keypoint', return_loss=True)
            keypoint_losses = self.keypoint_head.loss(
                *loss_inputs,
                gt_bboxes_ignore=gt_bboxes_ignore,
                gt_masks_ignore=gt_masks_ignore,
                gt_keypoints_ignore=gt_keypoints_ignore)
            all_losses.update(keypoint_losses)
        if self.with_auto_loss:
            all_losses = self.group_losses(all_losses)
            all_losses = self.loss_auto_weighted(*all_losses)
        return all_losses

    def group_losses(self, losses):
        return losses