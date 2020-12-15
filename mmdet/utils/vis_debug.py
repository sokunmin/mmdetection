import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

from mmdet.core import BitmapMasks
from mmdet.core.bbox.transforms import mask2polybox


def show_anno(img, gt_bboxes, gt_masks, gt_keypoints, kp_skeleton='coco', save_img=False):
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = img.astype(np.uint16) + 1

    if kp_skeleton != 'coco':
        raise NotImplementedError
    else:
        skeleton = np.array([[15, 13], [13, 11], [16, 14], [14, 12],
                             [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
                             [6, 8], [7, 9], [8, 10], [1, 2], [0, 1],
                             [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]])
    gt_mask_polys = []
    masks = gt_masks.masks if isinstance(gt_masks, BitmapMasks) else gt_masks
    if gt_bboxes is None:
        gt_bboxes = []
        for mask in masks:
            polys, bbox = mask2polybox(mask)
            gt_bboxes.append(bbox)
            gt_mask_polys.append(polys)
    else:
        for mask in masks:
            polys, _ = mask2polybox(mask)
            gt_mask_polys.append(polys)

    num_instances = len(gt_bboxes)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.imshow(img)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    plt.text(10, 30, f'#obj={num_instances}', fontsize=14, color='r')

    for i in range(num_instances):
        # > bbox drawing
        text_pos = {}
        if np.size(gt_bboxes) > 0 and type(gt_bboxes[i]) in (np.ndarray, list):
            c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
            bbox = gt_bboxes[i]
            [x1, y1, x2, y2] = bbox
            poly = [[x1, y1], [x1, y2],
                    [x2, y2], [x2, y1]]
            np_poly = np.array(poly).reshape((4, 2))

            polygons.append(Polygon(np_poly))
            color.append(c)

            text_pos['pos'] = [x1, y1]

        # > mask drawing
        if np.size(gt_mask_polys) > 0 and type(gt_mask_polys[i]) in (np.ndarray, list):
            c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
            # one instance is likely to have multiple polygons
            for poly in gt_mask_polys[i]:
                polygons.append(Polygon(poly))
                color.append(c)

        # > keypoint drawing
        if np.size(gt_keypoints) > 0 and type(gt_keypoints[i]) in (np.ndarray, list):
            c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
            # turn skeleton into zero-based index
            sks = skeleton
            kps = gt_keypoints[i]
            x = kps[:, 0]
            y = kps[:, 1]
            v = kps[:, 2]
            if kp_skeleton == 'cmu':  # map back to coco type
                map_v = {2: 0, 0: 1, 1: 2, 3: 0}
                v = np.vectorize(map_v.get)(v)
            text_pos['num_kps'] = sum(v > 0)
            for sk in sks:
                if np.all(v[sk] > 0):
                    plt.plot(x[sk], y[sk], linewidth=2, color=c)
            # > `coco -> cmu`: not labeled={0:2}, labeled={1:0, 2:1}
            plt.plot(x[v > 0], y[v > 0], 'o', markersize=7, markerfacecolor=c, markeredgecolor='k',
                     markeredgewidth=1)

        if bool(text_pos):
            pos = text_pos['pos']
            num_kps = int(text_pos['num_kps'])
            plt.text(pos[0], pos[1] + 30, f'[{i}]k={num_kps}', fontsize=12, color='b')

    p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
    ax.add_collection(p)
    p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
    ax.add_collection(p)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if save_img:
        plt.imsave('ann_resize_anno.jpg', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()