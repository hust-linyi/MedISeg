
from unittest import result
from skimage.measure import label
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score, precision_score
from scipy.spatial.distance import directed_hausdorff as hausdorff
from scipy.ndimage.measurements import center_of_mass
from sklearn.metrics import jaccard_score
from os.path import join
import os
import numpy as np


def compute_metrics(pred, gt, names):
    """
    Computes metrics specified by names between predicted label and groundtruth label.
    """

    gt_labeled = label(gt)
    pred_labeled = label(pred)

    gt_binary = gt_labeled.copy()
    pred_binary = pred_labeled.copy()
    gt_binary[gt_binary > 0] = 1
    pred_binary[pred_binary > 0] = 1
    gt_binary, pred_binary = gt_binary.flatten(), pred_binary.flatten()

    results = {}

    # pixel-level metrics
    if 'acc' in names:
        results['acc'] = accuracy_score(gt_binary, pred_binary)
    if 'roc' in names:
        results['roc'] = roc_auc_score(gt_binary, pred_binary)
    if 'p_F1' in names:  # pixel-level F1
        results['p_F1'] = f1_score(gt_binary, pred_binary)
    if 'p_recall' in names:  # pixel-level F1
        results['p_recall'] = recall_score(gt_binary, pred_binary)
    if 'p_precision' in names:  # pixel-level F1
        results['p_precision'] = precision_score(gt_binary, pred_binary)

    # object-level metrics
    if 'aji' in names:
        results['aji'] = AJI_fast(gt_labeled, pred_labeled)
    if 'haus' in names:
        results['dice'], results['iou'], results['haus'] = accuracy_object_level(pred_labeled, gt_labeled, True)
    if 'miou' in names:
        results['miou'] = jaccard_score(gt_binary, pred_binary, average='macro')
    if 'dice' in names:
        # results['dice'], results['iou'], _ = accuracy_object_level(pred_labeled, gt_labeled, False)
        results['dice'] = get_dice_1(gt_labeled, pred_labeled)
    

    return results


def accuracy_object_level(pred, gt, hausdorff_flag=True):
    """ Compute the object-level metrics between predicted and
    groundtruth: dice, iou, hausdorff """
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    if not isinstance(gt, np.ndarray):
        gt = np.array(gt)

    # get connected components
    pred_labeled = label(pred, connectivity=2)
    Ns = len(np.unique(pred_labeled)) - 1
    gt_labeled = label(gt, connectivity=2)
    Ng = len(np.unique(gt_labeled)) - 1

    # --- compute dice, iou, hausdorff --- #
    pred_objs_area = np.sum(pred_labeled>0)  # total area of objects in image
    gt_objs_area = np.sum(gt_labeled>0)  # total area of objects in groundtruth gt

    # compute how well groundtruth object overlaps its segmented object
    dice_g = 0.0
    iou_g = 0.0
    hausdorff_g = 0.0
    for i in range(1, Ng + 1):
        gt_i = np.where(gt_labeled == i, 1, 0)
        overlap_parts = gt_i * pred_labeled

        # get intersection objects numbers in image
        obj_no = np.unique(overlap_parts)
        obj_no = obj_no[obj_no != 0]

        gamma_i = float(np.sum(gt_i)) / gt_objs_area

        # show_figures((pred_labeled, gt_i, overlap_parts))

        if obj_no.size == 0:   # no intersection object
            dice_i = 0
            iou_i = 0

            # find nearest segmented object in hausdorff distance
            if hausdorff_flag:
                min_haus = 1e3

                # find overlap object in a window [-50, 50]
                pred_cand_indices = find_candidates(gt_i, pred_labeled)

                for j in pred_cand_indices:
                    pred_j = np.where(pred_labeled == j, 1, 0)
                    seg_ind = np.argwhere(pred_j)
                    gt_ind = np.argwhere(gt_i)
                    haus_tmp = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])

                    if haus_tmp < min_haus:
                        min_haus = haus_tmp
                haus_i = min_haus
        else:
            # find max overlap object
            obj_areas = [np.sum(overlap_parts == k) for k in obj_no]
            seg_obj = obj_no[np.argmax(obj_areas)]  # segmented object number
            pred_i = np.where(pred_labeled == seg_obj, 1, 0)  # segmented object

            overlap_area = np.max(obj_areas)  # overlap area

            dice_i = 2 * float(overlap_area) / (np.sum(pred_i) + np.sum(gt_i))
            iou_i = float(overlap_area) / (np.sum(pred_i) + np.sum(gt_i) - overlap_area)

            # compute hausdorff distance
            if hausdorff_flag:
                seg_ind = np.argwhere(pred_i)
                gt_ind = np.argwhere(gt_i)
                haus_i = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])

        dice_g += gamma_i * dice_i
        iou_g += gamma_i * iou_i
        if hausdorff_flag:
            hausdorff_g += gamma_i * haus_i

    # compute how well segmented object overlaps its groundtruth object
    dice_s = 0.0
    iou_s = 0.0
    hausdorff_s = 0.0
    for j in range(1, Ns + 1):
        pred_j = np.where(pred_labeled == j, 1, 0)
        overlap_parts = pred_j * gt_labeled

        # get intersection objects number in gt
        obj_no = np.unique(overlap_parts)
        obj_no = obj_no[obj_no != 0]

        # show_figures((pred_j, gt_labeled, overlap_parts))

        sigma_j = float(np.sum(pred_j)) / pred_objs_area
        # no intersection object
        if obj_no.size == 0:
            dice_j = 0
            iou_j = 0

            # find nearest groundtruth object in hausdorff distance
            if hausdorff_flag:
                min_haus = 1e3

                # find overlap object in a window [-50, 50]
                gt_cand_indices = find_candidates(pred_j, gt_labeled)

                for i in gt_cand_indices:
                    gt_i = np.where(gt_labeled == i, 1, 0)
                    seg_ind = np.argwhere(pred_j)
                    gt_ind = np.argwhere(gt_i)
                    haus_tmp = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])

                    if haus_tmp < min_haus:
                        min_haus = haus_tmp
                haus_j = min_haus
        else:
            # find max overlap gt
            gt_areas = [np.sum(overlap_parts == k) for k in obj_no]
            gt_obj = obj_no[np.argmax(gt_areas)]  # groundtruth object number
            gt_j = np.where(gt_labeled == gt_obj, 1, 0)  # groundtruth object

            overlap_area = np.max(gt_areas)  # overlap area

            dice_j = 2 * float(overlap_area) / (np.sum(pred_j) + np.sum(gt_j))
            iou_j = float(overlap_area) / (np.sum(pred_j) + np.sum(gt_j) - overlap_area)

            # compute hausdorff distance
            if hausdorff_flag:
                seg_ind = np.argwhere(pred_j)
                gt_ind = np.argwhere(gt_j)
                haus_j = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])

        dice_s += sigma_j * dice_j
        iou_s += sigma_j * iou_j
        if hausdorff_flag:
            hausdorff_s += sigma_j * haus_j

    return (dice_g + dice_s) / 2, (iou_g + iou_s) / 2, (hausdorff_g + hausdorff_s) / 2


def find_candidates(obj_i, objects_labeled, radius=50):
    """
    find object indices in objects_labeled in a window centered at obj_i
    when computing object-level hausdorff distance

    """
    if radius > 400:
        return np.array([])

    h, w = objects_labeled.shape
    x, y = center_of_mass(obj_i)
    x, y = int(x), int(y)
    r1 = x-radius if x-radius >= 0 else 0
    r2 = x+radius if x+radius <= h else h
    c1 = y-radius if y-radius >= 0 else 0
    c2 = y+radius if y+radius < w else w
    indices = np.unique(objects_labeled[r1:r2, c1:c2])
    indices = indices[indices != 0]

    if indices.size == 0:
        indices = find_candidates(obj_i, objects_labeled, 2*radius)

    return indices


def AJI_fast(gt, pred_arr):
    gs, g_areas = np.unique(gt, return_counts=True)
    assert np.all(gs == np.arange(len(gs)))
    ss, s_areas = np.unique(pred_arr, return_counts=True)
    assert np.all(ss == np.arange(len(ss)))

    i_idx, i_cnt = np.unique(np.concatenate([gt.reshape(1, -1), pred_arr.reshape(1, -1)]),
                             return_counts=True, axis=1)
    i_arr = np.zeros(shape=(len(gs), len(ss)), dtype=np.int)
    i_arr[i_idx[0], i_idx[1]] += i_cnt
    u_arr = g_areas.reshape(-1, 1) + s_areas.reshape(1, -1) - i_arr
    iou_arr = 1.0 * i_arr / u_arr

    i_arr = i_arr[1:, 1:]
    u_arr = u_arr[1:, 1:]
    iou_arr = iou_arr[1:, 1:]

    j = np.argmax(iou_arr, axis=1)

    c = np.sum(i_arr[np.arange(len(gs) - 1), j])
    u = np.sum(u_arr[np.arange(len(gs) - 1), j])
    used = np.zeros(shape=(len(ss) - 1), dtype=np.int)
    used[j] = 1
    u += (np.sum(s_areas[1:] * (1 - used)))
    return 1.0 * c / u


def get_dice_1(true, pred):
    """Traditional dice."""
    # cast to binary 1st
    true = np.copy(true)
    pred = np.copy(pred)
    true[true > 0] = 1
    pred[pred > 0] = 1
    inter = true * pred
    denom = true + pred
    return 2.0 * np.sum(inter) / np.sum(denom)

