import cv2
import mmcv
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from mmcv import imshow

from mmcv.image import imread, imwrite
from mmdet.core import bbox2result
from mmdet.models import build_loss
from .base import BaseDetector
from ..builder import DETECTORS, build_backbone, build_head, build_neck
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
                 loss_balance=None,
                 show_tb_debug=False):
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
        self.with_balance_loss = loss_balance is not None
        if self.with_balance_loss:
            self.loss_balance = build_loss(loss_balance)
        self.show_tb_debug = show_tb_debug
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
            x, all_outs = self.mask_head.preprocess(x, all_outs)
            all_outs['mask'] = self.mask_head(x)
        if self.with_keypoint:
            x, all_outs = self.keypoint_head.preprocess(x, all_outs)
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
        if isinstance(x, tuple):
            feat_shapes = [tuple(lvl.size()[2:]) for lvl in x]
        else:
            feat_shapes = [tuple(x.size()[2:])]
        if gt_masks is not None:
            gt_masks = [
                gt_mask.to_tensor(dtype=torch.uint8, device=img.device)
                for gt_mask in gt_masks
            ]
        gt_inputs = (gt_bboxes, gt_masks, gt_keypoints, gt_labels)
        all_targets = dict(bbox=None, mask=None, keypoint=None)
        all_metas = dict(img=img_metas, bbox=None, mask=None, keypoint=None)
        all_preds, all_losses = {}, {}
        if self.with_bbox:
            x, all_preds = self.bbox_head.preprocess(x, all_preds, training=True)
            all_preds['bbox'] = self.bbox_head(x)
            bbox_gts = self.bbox_head.interprocess(
                all_preds, gt_inputs, all_metas, training=True)
            all_targets['bbox'], all_metas['bbox'] = self.bbox_head.get_targets(
                *bbox_gts, feat_shapes=feat_shapes, img_metas=img_metas)
            loss_inputs = self.bbox_head.postprocess(
                all_preds['bbox'], all_targets, all_metas, training=True)
            bbox_losses = self.bbox_head.loss(
                *loss_inputs,
                img_metas=img_metas,
                gt_bboxes_ignore=gt_bboxes_ignore,
                gt_masks_ignore=gt_masks_ignore,
                gt_keypoints_ignore=gt_keypoints_ignore)
            all_losses.update(bbox_losses)
        if self.with_mask:
            x, all_preds = self.mask_head.preprocess(x, all_preds, training=True)
            all_preds['mask'] = self.mask_head(x)
            mask_gts = self.mask_head.interprocess(
                all_preds, gt_inputs, all_metas, training=True)
            all_targets['mask'], all_metas['mask'] = self.mask_head.get_targets(
                *mask_gts, feat_shapes=feat_shapes, img_metas=img_metas)
            loss_inputs = self.mask_head.postprocess(
                all_preds['mask'], all_targets, all_metas, training=True)
            mask_losses = self.mask_head.loss(
                *loss_inputs,
                img_metas=img_metas,
                gt_bboxes_ignore=gt_bboxes_ignore,
                gt_masks_ignore=gt_masks_ignore,
                gt_keypoints_ignore=gt_keypoints_ignore)
            all_losses.update(mask_losses)
        if self.with_keypoint:
            x, all_preds = self.keypoint_head.preprocess(x, all_preds, training=True)
            all_preds['keypoint'] = self.keypoint_head(x)
            keypoint_gts = self.keypoint_head.interprocess(
                all_preds, gt_inputs, all_metas, training=True)
            all_targets['keypoint'], all_metas['keypoint'] = self.keypoint_head.get_targets(
                *keypoint_gts, feat_shapes=feat_shapes, img_metas=img_metas)
            loss_inputs = self.keypoint_head.postprocess(
                all_preds['keypoint'], all_targets, all_metas, training=True)
            keypoint_losses = self.keypoint_head.loss(
                *loss_inputs,
                img_metas=img_metas,
                gt_bboxes_ignore=gt_bboxes_ignore,
                gt_masks_ignore=gt_masks_ignore,
                gt_keypoints_ignore=gt_keypoints_ignore)
            all_losses.update(keypoint_losses)
        if self.with_balance_loss:
            all_losses = self.group_losses(all_losses)
            all_losses = self.loss_balance(*all_losses)
        if self.show_tb_debug:
            debug_results = self.debug_train(img, all_preds, all_targets, all_metas, gt_inputs)
            return all_losses, debug_results
        return all_losses

    def group_losses(self, losses):
        return losses

    def debug_train(self, imgs, all_preds, all_targets, all_metas, gt_inputs, show_results=False):
        pass

    def debug_test(self, imgs, all_preds, all_metas, show_results=False):
        pass

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
        all_preds = {}
        all_metas = dict(img=img_metas, bbox=None, mask=None, keypoint=None)
        all_results = dict(bbox=[[]], mask=[[]], keypoint=[[]])
        if self.with_bbox:
            x, all_preds = self.bbox_head.preprocess(x, all_preds)
            all_preds['bbox'] = self.bbox_head(x)
            bbox_outs = self.bbox_head.interprocess(all_preds, all_metas)
            bbox_dets, all_metas['bbox'] = self.bbox_head.get_bboxes(
                *bbox_outs, img_metas=img_metas, rescale=rescale, with_nms=True)
            all_preds, all_metas, bbox_dets = self.bbox_head.postprocess(
                all_preds, all_metas, bbox_dets)
            # skip post-processing when exporting to ONNX
            if not torch.onnx.is_in_onnx_export():
                all_results['bbox'] = [
                    # convert detection results to a list of numpy arrays.
                    bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                    for det_bboxes, det_labels in bbox_dets
                ]
        if self.with_mask:
            x, all_preds = self.mask_head.preprocess(x, all_preds)
            all_preds['mask'] = self.mask_head(x)
            mask_outs = self.mask_head.interprocess(all_preds, all_metas)
            mask_dets, all_metas['mask'] = self.mask_head.get_segm_masks(
                *mask_outs, img_metas=img_metas, rescale=rescale, with_nms=True)
            all_preds, all_metas, mask_dets = self.mask_head.postprocess(
                all_preds, all_metas, mask_dets)
            if not torch.onnx.is_in_onnx_export():
                all_results['mask'] = mask_dets
        if self.with_keypoint:
            x, all_preds = self.keypoint_head.preprocess(x, all_preds)
            all_preds['keypoint'] = self.keypoint_head(x)
            keypoint_outs = self.keypoint_head.interprocess(all_preds, all_metas)
            keypoint_dets, all_metas['keypoint'] = self.keypoint_head.get_keypoints(
                *keypoint_outs, img_metas=img_metas, rescale=rescale, with_nms=True)
            all_preds, all_metas, keypoint_dets = self.keypoint_head.postprocess(
                all_preds, all_metas, keypoint_dets)
            if not torch.onnx.is_in_onnx_export():
                all_results['keypoint'] = [
                    keypoint2result(keypoints, self.keypoint_head.num_classes)
                    for keypoints in keypoint_dets
                ]
        # > remove empty key/values
        result_lists = {k: v for k, v in all_results.items() if v}
        # > unpack values -> zip -> map tuple to list -> list
        if len(result_lists.keys()) == 1:  # > #tasks
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

    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    color_type='Set2',  # > 'hls, husl, pastel, Paired, Set2
                    num_colors=25,
                    thickness=1,
                    font_scale=0.5,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):

        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple) and len(result) > 1:  # > multi-task
            assert len(result) == 3  # > results must include bboxes, masks and keypoints
            bbox_result, segm_result, keypoint_result = result
        else:
            bbox_result, segm_result, keypoint_result = result, None, None
        bboxes = np.vstack(bbox_result)
        bbox_keep = bboxes[:, -1] > score_thr
        num_objs = bbox_keep.sum()
        colors = (np.array(sns.color_palette(color_type, num_colors)) * 255).astype(np.uint16).tolist()
        num_colors = len(colors)
        if num_objs == 0:
            return img
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        num_labels = max(labels) + 1
        # draw segmentation masks
        if segm_result and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            inds = np.where(bboxes[:, -1] > score_thr)[0]
            if num_labels > 1:
                np.random.seed(42)
                color_masks = [
                    np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                    for _ in range(max(labels) + 1)
                ]

            for i in inds:
                if num_labels > 1:
                    color_mask = color_masks[labels[i] % num_colors]
                else:
                    color_mask = np.array([colors[int(i) % num_colors]])
                mask = segms[i].astype(bool)
                img[mask] = img[mask] * 0.5 + color_mask * 0.5

        if keypoint_result:
            num_joints = self.keypoint_head.num_classes
            keypoints = np.vstack(keypoint_result)  # (B, K, ((xyv) * #kp)
            kp_shape = keypoints.shape
            joints = keypoints[bbox_keep].reshape(num_objs, kp_shape[1] // 3, 3)
            limbs = self.keypoint_head.limbs
            for i, joint in zip(range(num_objs), joints):  # > #objs
                color = colors[i % num_colors]
                for l, limb_ind in enumerate(limbs):  # `limbs`: (#kp, 2)
                    if all(joint[limb_ind, 2] > 0.):  # joints on a limb are above
                        src_pt = tuple(np.round(joint[limb_ind, :2][0]).astype(np.int32))
                        dst_pt = tuple(np.round(joint[limb_ind, :2][1]).astype(np.int32))
                        cv2.line(img, src_pt, dst_pt, color, thickness=2, lineType=cv2.LINE_AA)

                for j in range(num_joints):
                    if joint[j][2] > 0.:
                        cv2.circle(img, (joint[j, 0], joint[j, 1]), 2, mmcv.color_val('white'),
                                   thickness=cv2.FILLED, lineType=cv2.LINE_AA)

        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        assert bboxes.ndim == 2
        assert labels.ndim == 1
        assert bboxes.shape[0] == labels.shape[0]
        assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
        img = imread(img)

        if score_thr > 0:
            assert bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]

        img = np.ascontiguousarray(img)
        for i, bbox, label in zip(range(num_objs), bboxes, labels):
            color = colors[i % num_colors]
            bbox_int = bbox.astype(np.int32)
            left_top = (bbox_int[0], bbox_int[1])
            right_bottom = (bbox_int[2], bbox_int[3])
            cv2.rectangle(img, left_top, right_bottom, color, thickness=thickness)
            label_text = self.CLASSES[label] if self.CLASSES is not None else f'cls {label}'
            if len(bbox) > 4:
                label_text += f'|{bbox[-1]:.02f}'
            cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                        cv2.FONT_HERSHEY_COMPLEX, font_scale, color)

        if show:
            imshow(img, win_name, wait_time)
        if out_file is not None:
            imwrite(img, out_file)

        if not (show or out_file):
            return img