import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import pandas as pd
from collections import OrderedDict
import glob
from utils.util import AverageMeterArray
from sklearn.metrics import recall_score, precision_score, f1_score, jaccard_score


def test_all_case(net, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, save_result=True, test_save_path=None):
    metric_names = ['recall1', 'precision1', 'dice1', 'miou1', 'recall2', 'precision2', 'dice2', 'miou2']
    total_metric = AverageMeterArray(len(metric_names))
    if save_result:
        if not os.path.exists(test_save_path + '/img'):
            os.makedirs(test_save_path + '/img')
    for image_path in tqdm(image_list):
        case_name = os.path.basename(image_path).replace('_image.npy', '')
        image = np.load(image_path)
        label = np.load(image_path.replace('image', 'label'))
        image = np.squeeze(image)
        label = np.squeeze(label)
        prediction, score_map = test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)

        single_metric = calculate_metric_percase(prediction, label)
        total_metric.update([single_metric[metric_name] for metric_name in metric_names])

        if save_result:
            # nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), os.path.join(test_save_path, 'img', case_name+'_pred.nii.gz'))
            # nib.save(nib.Nifti1Image(score_map.astype(np.float32), np.eye(4)), os.path.join(test_save_path, 'img', case_name+'_prob.nii.gz'))
            # nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), os.path.join(test_save_path, 'img', case_name+'_img.nii.gz'))
            # nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), os.path.join(test_save_path, 'img', case_name+'_gt.nii.gz'))
            np.save(os.path.join(test_save_path, 'img', case_name+'_pred.npy'), prediction)
            np.save(os.path.join(test_save_path, 'img', case_name+'_prob.npy'), score_map)
            np.save(os.path.join(test_save_path, 'img', case_name+'_img.npy'), image)
            np.save(os.path.join(test_save_path, 'img', case_name+'_gt.npy'), label)
    result_avg = [[total_metric.avg[i]*100 for i in range(len(metric_names))]]
    result_avg = pd.DataFrame(result_avg, columns=metric_names)
    result_avg.to_csv(os.path.join(test_save_path, 'test_results.csv'), index=False)
    return


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                y1 = net(test_patch)
                y = F.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0,:,:,:,:]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt,axis=0)
    label_map = np.argmax(score_map, axis = 0)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map

def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num-1)
    for i in range(1, num):
        prediction_tmp = (prediction==i)
        label_tmp = (label==i)
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice

    return total_dice


def calculate_metric_percase(pred, gt):
    # measurement: recall, precision, dice, miou
    # for kidney 1, and tumor 2
    pred1 = (pred > 0).astype(int).flatten()
    gt1 = (gt > 0).astype(int).flatten()
    pred2 = (pred == 2).astype(int).flatten()
    gt2 = (gt == 2).astype(int).flatten()

    result = {}
    if pred1.sum() == 0:
        result['recall1'] = 0
        result['precision1'] = 0
        result['dice1'] = 0
        result['miou1'] = 0
    else:
        result['recall1'] = metric.binary.recall(pred1, gt1)
        result['precision1'] = metric.binary.precision(pred1, gt1)
        result['dice1'] = metric.binary.dc(pred1, gt1)
        result['miou1'] = jaccard_score(gt1, pred1)
    if pred2.sum() == 0:
        result['recall2'] = 0
        result['precision2'] = 0
        result['dice2'] = 0
        result['miou2'] = 0
    else:
        result['recall2'] = metric.binary.recall(pred2, gt2)
        result['precision2'] = metric.binary.precision(pred2, gt2)
        result['dice2'] = metric.binary.dc(pred2, gt2)
        result['miou2'] = jaccard_score(gt2, pred2)

    return result


def get_imglist(root_path, fold):
    img_list = [f for f in os.listdir(root_path) if f.endswith('image.npy')]
    valnum = int(len(img_list) * 0.2)
    valstart = valnum * fold
    valend = valnum * (fold + 1)
    img_list = img_list[valstart:valend]
    img_list = [os.path.join(root_path, f) for f in img_list]
    return img_list