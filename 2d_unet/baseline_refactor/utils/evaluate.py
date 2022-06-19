# import numpy as np
# import pathlib
# import os
# import shutil

# from scipy.optimize import linear_sum_assignment
# from sklearn.metrics import r2_score


# def remap_label(pred, by_size=False):
#     """Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3] 
#     not [0, 2, 4, 6]. The ordering of instances (which one comes first) 
#     is preserved unless by_size=True, then the instances will be reordered
#     so that bigger nucler has smaller ID.

#     Args:
#         pred (ndarray): the 2d array contain instances where each instances is marked
#             by non-zero integer.
#         by_size (bool): renaming such that larger nuclei have a smaller id (on-top).

#     Returns:
#         new_pred (ndarray): Array with continguous ordering of instances.
        
#     """
#     pred_id = list(np.unique(pred))
#     pred_id.remove(0)
#     if len(pred_id) == 0:
#         return pred  # no label
#     if by_size:
#         pred_size = []
#         for inst_id in pred_id:
#             size = (pred == inst_id).sum()
#             pred_size.append(size)
#         # sort the id by size in descending order
#         pair_list = zip(pred_id, pred_size)
#         pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
#         pred_id, pred_size = zip(*pair_list)

#     new_pred = np.zeros(pred.shape, np.int32)
#     for idx, inst_id in enumerate(pred_id):
#         new_pred[pred == inst_id] = idx + 1
#     return new_pred


# def cropping_center(x, crop_shape, batch=False):
#     """Crop an array at the centre with specified dimensions."""
#     orig_shape = x.shape
#     if not batch:
#         h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
#         w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
#         x = x[h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
#     else:
#         h0 = int((orig_shape[1] - crop_shape[0]) * 0.5)
#         w0 = int((orig_shape[2] - crop_shape[1]) * 0.5)
#         x = x[:, h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
#     return x


# def recur_find_ext(root_dir, ext_list):
#     """Recursively find all files in directories end with the `ext` such as `ext='.png'`.
    
#     Args:
#         root_dir (str): Root directory to grab filepaths from.
#         ext_list (list): File extensions to consider.
        
#     Returns:
#         file_path_list (list): sorted list of filepaths.
#     """
#     file_path_list = []
#     for cur_path, dir_list, file_list in os.walk(root_dir):
#         for file_name in file_list:
#             file_ext = pathlib.Path(file_name).suffix
#             if file_ext in ext_list:
#                 full_path = os.path.join(cur_path, file_name)
#                 file_path_list.append(full_path)
#     file_path_list.sort()
#     return file_path_list


# def rm_n_mkdir(dir_path):
#     """Remove and then make a new directory."""
#     if os.path.isdir(dir_path):
#         shutil.rmtree(dir_path)
#     os.makedirs(dir_path)


# def get_bounding_box(img):
#     """Get the bounding box coordinates of a binary input- assumes a single object.
    
#     Args:
#         img: input binary image.
    
#     Returns:
#         bounding box coordinates
        
#     """
#     rows = np.any(img, axis=1)
#     cols = np.any(img, axis=0)
#     rmin, rmax = np.where(rows)[0][[0, -1]]
#     cmin, cmax = np.where(cols)[0][[0, -1]]
#     # due to python indexing, need to add 1 to max
#     # else accessing will be 1px in the box, not out
#     rmax += 1
#     cmax += 1
#     return [rmin, rmax, cmin, cmax]


# def get_multi_pq_info(true, pred, nr_classes=6, match_iou=0.5):
#     """Get the statistical information needed to compute multi-class PQ.
    
#     CoNIC multiclass PQ is achieved by considering nuclei over all images at the same time, 
#     rather than averaging image-level results, like was done in MoNuSAC. This overcomes issues
#     when a nuclear category is not present in a particular image.
    
#     Args:
#         true (ndarray): HxWx2 array. First channel is the instance segmentation map
#             and the second channel is the classification map. 
#         pred: HxWx2 array. First channel is the instance segmentation map
#             and the second channel is the classification map. 
#         nr_classes (int): Number of classes considered in the dataset. 
#         match_iou (float): IoU threshold for determining whether there is a detection.
    
#     Returns:
#         statistical info per class needed to compute PQ.
    
#     """

#     assert match_iou >= 0.0, "Cant' be negative"

#     true_inst = true[..., 0]
#     pred_inst = pred[..., 0]
#     ###
#     true_class = true[..., 1]
#     pred_class = pred[..., 1]

#     pq = []
#     for idx in range(nr_classes):
#         pred_class_tmp = pred_class == idx + 1
#         pred_inst_oneclass = pred_inst * pred_class_tmp
#         pred_inst_oneclass = remap_label(pred_inst_oneclass)
#         ##
#         true_class_tmp = true_class == idx + 1
#         true_inst_oneclass = true_inst * true_class_tmp
#         true_inst_oneclass = remap_label(true_inst_oneclass)

#         pq_oneclass_info = get_pq(true_inst_oneclass, pred_inst_oneclass, remap=False)

#         # add (in this order) tp, fp, fn iou_sum
#         pq_oneclass_stats = [
#             pq_oneclass_info[1][0],
#             pq_oneclass_info[1][1],
#             pq_oneclass_info[1][2],
#             pq_oneclass_info[2],
#         ]
#         pq.append(pq_oneclass_stats)

#     return pq


# def get_pq(true, pred, match_iou=0.5, remap=True):
#     """Get the panoptic quality result. 
    
#     Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4] 
#     not [2, 3, 6, 10]. Please call `remap_label` beforehand. Here, the `by_size` flag 
#     has no effect on the result.

#     Args:
#         true (ndarray): HxW ground truth instance segmentation map
#         pred (ndarray): HxW predicted instance segmentation map
#         match_iou (float): IoU threshold level to determine the pairing between
#             GT instances `p` and prediction instances `g`. `p` and `g` is a pair
#             if IoU > `match_iou`. However, pair of `p` and `g` must be unique 
#             (1 prediction instance to 1 GT instance mapping). If `match_iou` < 0.5, 
#             Munkres assignment (solving minimum weight matching in bipartite graphs) 
#             is caculated to find the maximal amount of unique pairing. If 
#             `match_iou` >= 0.5, all IoU(p,g) > 0.5 pairing is proven to be unique and
#             the number of pairs is also maximal.  
#         remap (bool): whether to ensure contiguous ordering of instances.
    
#     Returns:
#         [dq, sq, pq]: measurement statistic

#         [paired_true, paired_pred, unpaired_true, unpaired_pred]: 
#                       pairing information to perform measurement
        
#         paired_iou.sum(): sum of IoU within true positive predictions
                    
#     """
#     assert match_iou >= 0.0, "Cant' be negative"
#     # ensure instance maps are contiguous
#     if remap:
#         pred = remap_label(pred)
#         true = remap_label(true)

#     true = np.copy(true)
#     pred = np.copy(pred)
#     true = true.astype("int32")
#     pred = pred.astype("int32")
#     true_id_list = list(np.unique(true))
#     pred_id_list = list(np.unique(pred))
#     # prefill with value
#     pairwise_iou = np.zeros([len(true_id_list), len(pred_id_list)], dtype=np.float64)

#     # caching pairwise iou
#     for true_id in true_id_list[1:]:  # 0-th is background
#         t_mask_lab = true == true_id
#         rmin1, rmax1, cmin1, cmax1 = get_bounding_box(t_mask_lab)
#         t_mask_crop = t_mask_lab[rmin1:rmax1, cmin1:cmax1]
#         t_mask_crop = t_mask_crop.astype("int")
#         p_mask_crop = pred[rmin1:rmax1, cmin1:cmax1]
#         pred_true_overlap = p_mask_crop[t_mask_crop > 0]
#         pred_true_overlap_id = np.unique(pred_true_overlap)
#         pred_true_overlap_id = list(pred_true_overlap_id)
#         for pred_id in pred_true_overlap_id:
#             if pred_id == 0:  # ignore
#                 continue  # overlaping background
#             p_mask_lab = pred == pred_id
#             p_mask_lab = p_mask_lab.astype("int")

#             # crop region to speed up computation
#             rmin2, rmax2, cmin2, cmax2 = get_bounding_box(p_mask_lab)
#             rmin = min(rmin1, rmin2)
#             rmax = max(rmax1, rmax2)
#             cmin = min(cmin1, cmin2)
#             cmax = max(cmax1, cmax2)
#             t_mask_crop2 = t_mask_lab[rmin:rmax, cmin:cmax]
#             p_mask_crop2 = p_mask_lab[rmin:rmax, cmin:cmax]

#             total = (t_mask_crop2 + p_mask_crop2).sum()
#             inter = (t_mask_crop2 * p_mask_crop2).sum()
#             iou = inter / (total - inter)
#             pairwise_iou[true_id - 1, pred_id - 1] = iou

#     if match_iou >= 0.5:
#         paired_iou = pairwise_iou[pairwise_iou > match_iou]
#         pairwise_iou[pairwise_iou <= match_iou] = 0.0
#         paired_true, paired_pred = np.nonzero(pairwise_iou)
#         paired_iou = pairwise_iou[paired_true, paired_pred]
#         paired_true += 1  # index is instance id - 1
#         paired_pred += 1  # hence return back to original
#     else:  # * Exhaustive maximal unique pairing
#         #### Munkres pairing with scipy library
#         # the algorithm return (row indices, matched column indices)
#         # if there is multiple same cost in a row, index of first occurence
#         # is return, thus the unique pairing is ensure
#         # inverse pair to get high IoU as minimum
#         paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
#         ### extract the paired cost and remove invalid pair
#         paired_iou = pairwise_iou[paired_true, paired_pred]

#         # now select those above threshold level
#         # paired with iou = 0.0 i.e no intersection => FP or FN
#         paired_true = list(paired_true[paired_iou > match_iou] + 1)
#         paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
#         paired_iou = paired_iou[paired_iou > match_iou]

#     # get the actual FP and FN
#     unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
#     unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]
#     # print(paired_iou.shape, paired_true.shape, len(unpaired_true), len(unpaired_pred))

#     #
#     tp = len(paired_true)
#     fp = len(unpaired_pred)
#     fn = len(unpaired_true)
#     # get the F1-score i.e DQ
#     dq = tp / ((tp + 0.5 * fp + 0.5 * fn) + 1.0e-6)
#     # get the SQ, no paired has 0 iou so not impact
#     sq = paired_iou.sum() / (tp + 1.0e-6)

#     return (
#         [dq, sq, dq * sq],
#         [tp, fp, fn],
#         paired_iou.sum(),
#     )


# def get_multi_r2(true, pred):
#     """Get the correlation of determination for each class and then 
#     average the results.
    
#     Args:
#         true (pd.DataFrame): dataframe indicating the nuclei counts for each image and category.
#         pred (pd.DataFrame): dataframe indicating the nuclei counts for each image and category.
    
#     Returns:
#         multi class coefficient of determination
        
#     """
#     # first check to make sure that the appropriate column headers are there
#     class_names = [
#         "epithelial",
#         "lymphocyte",
#         "plasma",
#         "neutrophil",
#         "eosinophil",
#         "connective",
#     ]
#     for col in true.columns:
#         if col not in class_names:
#             raise ValueError("%s column header not recognised")

#     for col in pred.columns:
#         if col not in class_names:
#             raise ValueError("%s column header not recognised")

#     # for each class, calculate r2 and then take the average
#     r2_list = []
#     for class_ in class_names:
#         true_oneclass = true[class_].tolist()
#         pred_oneclass = pred[class_].tolist()
#         r2_list.append(r2_score(true_oneclass, pred_oneclass))

#     return np.mean(np.array(r2_list))