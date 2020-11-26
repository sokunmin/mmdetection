import torch
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
                 loss_balance=None):
        super(SingleStageMultiDetector, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        if bbox_head is not None:
            bbox_head.update(train_cfg=train_cfg)
            bbox_head.update(test_cfg=test_cfg)
            self.bbox_head = build_head(bbox_head)
        if mask_head is not None:
            mask_head.update(train_cfg=train_cfg)
            mask_head.update(test_cfg=test_cfg)
            self.mask_head = build_head(mask_head)
        if keypoint_head is not None:
            keypoint_head.update(train_cfg=train_cfg)
            keypoint_head.update(test_cfg=test_cfg)
            self.keypoint_head = build_head(keypoint_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if loss_balance is not None:
            self.loss_balance = build_loss(loss_balance)
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

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        all_outs = {}
        if self.with_bbox:
            x, all_outs = self.bbox_head.preprocess(x, all_outs)
            all_outs['bbox'] = self.bbox_head(x)

        if self.with_mask:
            x, all_outs = self.group_head_inputs(x, all_outs)
            all_outs['mask'] = self.mask_head(x)
        if self.with_keypoint:
            x, all_outs = self.group_head_inputs(x, all_outs)
            all_outs['keypoint'] = self.keypoint_head(x)
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
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x = self.extract_feat(img)
        all_outs = {}
        if self.with_bbox:
            x, all_outs = self.bbox_head.preprocess(x, all_outs)
            all_outs['bbox'] = self.bbox_head(x)
            all_outs = self.bbox_head.postprocess(all_outs, is_train=True)
        if self.with_mask:
            x, all_outs = self.mask_head.preprocess(x, all_outs)
            all_outs['mask'] = self.mask_head(x)
            all_outs = self.mask_head.postprocess(all_outs, is_train=True)
        if self.with_keypoint:
            x, all_outs = self.keypoint_head.preprocess(x, all_outs)
            all_outs['keypoint'] = self.keypoint_head(x)
            all_outs = self.keypoint_head.postprocess(all_outs, is_train=True)

        gt_inputs = (gt_bboxes, gt_masks, gt_keypoints, gt_labels)
        losses = self.loss(all_outs, gt_inputs, img_metas,
                           gt_bboxes_ignore=gt_bboxes_ignore,
                           gt_masks_ignore=gt_masks_ignore,
                           gt_keypoints_ignore=gt_keypoints_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        all_outs, all_results = {}, {'bbox': [], 'mask': [], 'keypoint': []}
        if self.with_bbox:
            x, all_outs = self.bbox_head.preprocess(x, all_outs)
            all_outs['bbox'] = self.bbox_head(x)
            # > (n, [bboxes, labels, ...])
            bbox_list = self.bbox_head.get_bboxes(*all_outs.values(), img_metas, rescale=rescale)
            all_outs, bbox_list = self.bbox_head.postprocess(all_outs, bbox_list)
            # skip post-processing when exporting to ONNX
            if not torch.onnx.is_in_onnx_export():
                bbox_results = [
                    # convert detection results to a list of numpy arrays.
                    bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                    for det_bboxes, det_labels in bbox_list
                ]
                all_results['bbox'] = bbox_results
        if self.with_mask:
            x, all_outs = self.mask_head.preprocess(x, all_outs)
            all_outs['mask'] = self.mask_head(x)
            mask_list = self.mask_head.get_masks(*all_outs.values(), img_metas, rescale=rescale)
            all_outs, mask_list = self.mask_head.postprocess(all_outs, mask_list)
            if not torch.onnx.is_in_onnx_export():
                # TOCHECK: add `mask2result`
                all_results['mask'] = None
        if self.with_keypoint:
            x, all_outs = self.keypoint_head.preprocess(x, all_outs)
            all_outs['keypoint'] = self.keypoint_head(x)
            keypoint_list = self.keypoint_head.get_keypoints(*all_outs.values(), img_metas, rescale=rescale)
            all_outs, keypoint_list = self.keypoint_head.postprocess(all_outs, keypoint_list)
            if not torch.onnx.is_in_onnx_export():
                keypoint_results = [
                    keypoint2result(keypoints, self.keypoint_head.num_classes)
                    for keypoints in keypoint_list
                ]
                all_results['keypoint'] = keypoint_results
        # > remove empty key/values
        result_lists = {k: v for k, v in all_results.items() if v}
        # > unpack values -> zip -> map tuple to list -> list
        if len(result_lists.keys()) == 1:
            return list(result_lists.values())[0]
        return tuple(map(tuple, zip(*result_lists.values())))

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

    def group_losses(self, losses):
        return losses

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
            loss_inputs = self.bbox_head.postprocess(
                pred_outs['bbox'], gt_bboxes, gt_labels, img_metas,
                self.train_cfg, return_loss=True)
            bbox_losses = self.bbox_head.loss(
                *loss_inputs,
                gt_bboxes_ignore=gt_bboxes_ignore,
                gt_masks_ignore=gt_masks_ignore,
                gt_keypoints_ignore=gt_keypoints_ignore)
            all_losses.update(bbox_losses)
        if self.with_mask:
            loss_inputs = self.mask_head.postprocess(
                pred_outs['mask'], gt_masks, gt_labels, img_metas,
                self.train_cfg, return_loss=True)
            mask_losses = self.mask_head.loss(
                *loss_inputs,
                gt_bboxes_ignore=gt_bboxes_ignore,
                gt_masks_ignore=gt_masks_ignore,
                gt_keypoints_ignore=gt_keypoints_ignore)
            all_losses.update(mask_losses)
        if self.with_keypoint:
            loss_inputs = self.keypoint_head.postprocess(
                pred_outs['keypoint'], gt_keypoints, gt_labels, img_metas,
                self.train_cfg, return_loss=True)
            keypoint_losses = self.keypoint_head.loss(
                *loss_inputs,
                gt_bboxes_ignore=gt_bboxes_ignore,
                gt_masks_ignore=gt_masks_ignore,
                gt_keypoints_ignore=gt_keypoints_ignore)
            all_losses.update(keypoint_losses)
        if self.with_balance_loss:
            all_losses = self.group_losses(all_losses)
            all_losses = self.loss_balance(*all_losses)
        return all_losses