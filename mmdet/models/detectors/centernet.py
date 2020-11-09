import torch

import cv2
import mmcv
import numpy as np
from mmdet.core import bbox2result, keypoint2result
from ..builder import DETECTORS
from .single_stage import SingleStageDetector

try:
    import matplotlib
    import matplotlib.animation
    import matplotlib.collections
    import matplotlib.patches
    import matplotlib.cm as cm
except ImportError:
    matplotlib = None

@DETECTORS.register_module()
class CenterNet(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(CenterNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained)

    def simple_test(self, img, img_metas, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # TODO: split to `get_bboxes()`, `get_keypoints()`
        bbox_list, keypoint_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)

        # TODO: add `get_keypoints()`

        # TODO: add `group_predicts()`

        # skip post-processing when exporting to ONNX
        if torch.onnx.is_in_onnx_export():
            return bbox_list

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]

        keypoint_results = [
            keypoint2result(det_keypoints)
            for det_keypoints in keypoint_list
        ]
        # TODO: add `soft_nms39()`
        return bbox_results, keypoint_results

    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color='green',
                    text_color='green',
                    thickness=1,
                    font_scale=0.5,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):

        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, keypoint_result, segm_result = result
        else:
            bbox_result, keypoint_result, segm_result = result, None, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            inds = np.where(bboxes[:, -1] > score_thr)[0]
            np.random.seed(42)
            color_masks = [
                np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                for _ in range(max(labels) + 1)
            ]
            for i in inds:
                i = int(i)
                color_mask = color_masks[labels[i]]
                mask = segms[i].astype(bool)
                img[mask] = img[mask] * 0.5 + color_mask * 0.5

        if keypoint_result is not None:
            num_joints = self.bbox_head.num_keypoints
            joints = keypoint_result[:num_joints]
            joints = joints.astype(dtype=np.int32).view(num_joints, 2)  # (#kp, (y,x))
            joint_scores = keypoint_result[num_joints:]  # (#kp,)
            pose_colors = (np.array(cm.get_cmap('tab20').colors) * 255).astype(np.uint16).tolist()
            limbs = self.bbox_head.limbs
            for l, limb_ind in enumerate(limbs):  # `limbs`: (#kp, 2)
                if all(joint_scores[limb_ind] > 0.):  # joints on a limb are above
                    x = joints[limb_ind, 1]
                    y = joints[limb_ind, 0]
                    cv2.line(img, x, y, pose_colors[l], thickness=2, lineType=cv2.LINE_AA)

            for j in range(num_joints):
                if joint_scores[j] > 0.:
                    cv2.circle(img, (joints[j, 0], joints[j, 1]), 2, mmcv.color_val('white'),
                               thickness=cv2.FILLED, lineType=cv2.LINE_AA)

        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        mmcv.imshow_det_bboxes(
            img,
            bboxes,
            labels,
            class_names=self.CLASSES,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            thickness=thickness,
            font_scale=font_scale,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            return img