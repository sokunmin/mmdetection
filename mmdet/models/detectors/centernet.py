import torch

import cv2
import mmcv
import seaborn as sns
from mmcv import color_val, imshow
from mmcv.image import imread, imwrite
import numpy as np
from mmdet.core import bbox2result, bbox_mapping_back
from ..builder import DETECTORS
from .single_stage_mt import SingleStageMultiDetector

try:
    import matplotlib
    import matplotlib.animation
    import matplotlib.collections
    import matplotlib.patches
    import matplotlib.cm as cm
except ImportError:
    matplotlib = None


@DETECTORS.register_module()
class CenterNet(SingleStageMultiDetector):

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
        super(CenterNet, self).__init__(backbone, neck, bbox_head, mask_head, keypoint_head,
                                        train_cfg, test_cfg, pretrained, loss_balance)

    def merge_aug_results(self, aug_results, img_metas):
        """Merge augmented detection bboxes and score.

        Args:
            aug_results (list[list[Tensor]]): Det_bboxes and det_labels of each
                image.
            img_metas (list[list[dict]]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            tuple: (bboxes, labels)
        """
        recovered_bboxes, aug_labels = [], []
        for bboxes_labels, img_info in zip(aug_results, img_metas):
            img_shape = img_info[0]['img_shape']  # using shape before padding
            scale_factor = img_info[0]['scale_factor']
            flip = img_info[0]['flip']
            bboxes, labels = bboxes_labels
            bboxes, scores = bboxes[:, :4], bboxes[:, -1:]
            bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip)
            recovered_bboxes.append(torch.cat([bboxes, scores], dim=-1))
            aug_labels.append(labels)

        bboxes = torch.cat(recovered_bboxes, dim=0)
        labels = torch.cat(aug_labels)

        if bboxes.shape[0] > 0:
            out_bboxes, out_labels = self.bbox_head._bboxes_nms(
                bboxes, labels, self.bbox_head.test_cfg)
        else:
            out_bboxes, out_labels = bboxes, labels

        return out_bboxes, out_labels

    def aug_test(self, imgs, img_metas, rescale=False):
        """Augment testing of CornerNet.

        Args:
            imgs (list[Tensor]): Augmented images.
            img_metas (list[list[dict]]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Note:
            ``imgs`` must including flipped image pairs.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        img_inds = list(range(len(imgs)))

        assert img_metas[0][0]['flip'] + img_metas[1][0]['flip'], (
            'aug test must have flipped image pair')
        aug_results = []
        for ind, flip_ind in zip(img_inds[0::2], img_inds[1::2]):
            img_pair = torch.cat([imgs[ind], imgs[flip_ind]])
            x = self.extract_feat(img_pair)
            outs = self.bbox_head(x)
            bbox_list = self.bbox_head.get_bboxes(
                *outs, [img_metas[ind], img_metas[flip_ind]], False, False)
            aug_results.append(bbox_list[0])
            aug_results.append(bbox_list[1])

        bboxes, labels = self.merge_aug_results(aug_results, img_metas)
        bbox_results = bbox2result(bboxes, labels, self.bbox_head.num_classes)

        return [bbox_results]

    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    color_type='Paired',  # > 'hls, husl, Paired, Sets2
                    num_colors=20,
                    thickness=1,
                    font_scale=0.5,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):

        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple) and len(result) > 1:  # > multi-task
            if self.with_bbox and self.with_keypoint and not self.with_mask:
                bbox_result, keypoint_result = result
                segm_result = None
            elif self.with_bbox and self.with_mask and not self.with_keypoint:
                bbox_result, segm_result = result
                keypoint_result = None
            else:
                bbox_result, keypoint_result, segm_result = result
        else:
            bbox_result, keypoint_result, segm_result = result, None, None
        bboxes = np.vstack(bbox_result)
        bbox_keep = bboxes[:, -1] > score_thr
        num_objs = bbox_keep.sum()
        colors = (np.array(sns.color_palette(color_type, num_colors)) * 255).astype(np.uint16).tolist()  # 12 *
        num_colors = len(colors)
        if num_objs == 0:
            return img
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
            num_joints = self.keypoint_head.num_classes
            keypoints = np.vstack(keypoint_result)  # (B, K, ((xyv) * #kp + bbox_score)
            kp_shape = keypoints.shape
            joints = keypoints[bbox_keep, :-1].reshape(num_objs, kp_shape[1] // 3, 3).astype(dtype=np.int32)
            limbs = self.keypoint_head.limbs
            for i, joint in zip(range(num_objs), joints):  # > #objs
                color = colors[i % num_colors]
                for l, limb_ind in enumerate(limbs):  # `limbs`: (#kp, 2)
                    if all(joint[limb_ind, 2] > 0.):  # joints on a limb are above
                        src_pt = tuple(joint[limb_ind, :2][0])
                        dst_pt = tuple(joint[limb_ind, :2][1])
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