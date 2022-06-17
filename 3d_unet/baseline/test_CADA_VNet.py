import os
import re
import argparse
import torch
from networks.vnet import VNet
from networks.unet import UNet3D
import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch.nn.functional as F
import pandas as pd
from collections import OrderedDict
import SimpleITK as sitk
# import pdb


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/extracephonline/medai_data_hongyuzhou/ianylin/data/TB_CT/data_10.19/preprocess/data_clean/', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='unet_supervisedonly_dp', help='experiment_name')
parser.add_argument('--model', type=str,  default='unet_supervisedonly_dp', help='model_name')
parser.add_argument('--gpu', type=str,  default='4', help='GPU to use')
# parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
# parser.add_argument('--num_workers', type=int,  default=32, help='num_workers')
parser.add_argument('--num_classes', type=int,  default=10, help='num_classes')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = FLAGS.root_path + FLAGS.exp + "/"
data_path = FLAGS.root_path + 'preprocess/'
test_save_path = FLAGS.root_path + FLAGS.exp + "/test_raw"
test_save_path_2 = FLAGS.root_path + FLAGS.exp + "/test_post"
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
if not os.path.exists(test_save_path_2):
    os.makedirs(test_save_path_2)
with open(data_path+'val.txt', 'r') as f:
    image_list = f.readlines()
image_list = [item.replace('\n', '') for item in image_list]
# print(image_list)

def test_calculate_metric():
    # net = VNet(n_channels=1, n_classes=FLAGS.num_classes, normalization='instancenorm', has_dropout=False).cuda()
    net = UNet3D(num_classes=FLAGS.num_classes, input_channels=1, act='relu', norm='in').cuda()
    save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
    # save_mode_path = os.path.join(snapshot_path, 'checkpoint_1400.pth.tar')

    state_dict = torch.load(save_mode_path)['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    print("init weight from {}".format(save_mode_path))
    net.eval()

    # import pdb;pdb.set_trace()
    avg_metric = test_all_case(net, image_list, num_classes=FLAGS.num_classes,
                               patch_size=(96, 96, 96), stride_xy=64, stride_z=64,
                               save_result=True, test_save_path=test_save_path)

    return avg_metric



def test_all_case(net, image_list, num_classes, patch_size=(128, 128, 128), stride_xy=32, stride_z=32, save_result=True, test_save_path=None, preproc_fn=None):
    total_metric = 0.0
    metric_dict = OrderedDict()
    metric_dict['name'] = list()
    metric_dict['dice'] = list()
    metric_dict['jaccard'] = list()
    metric_dict['asd'] = list()
    metric_dict['95hd'] = list()
    for case_name in image_list:
        print(f'processing {case_name}...')
        image = np.load(os.path.join(data_path, case_name + "_image.npy"))
        label = np.load(os.path.join(data_path, case_name + "_label.npy"))

        # if preproc_fn is not None:
        #     image = preproc_fn(image)
        prediction, score_map = test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)

        if np.sum(prediction)==0:
            single_metric = (0,0,0,0)
            metric_dict['name'].append(case_name)
            metric_dict['dice'].append(0)
            metric_dict['jaccard'].append(0)
            metric_dict['asd'].append(0)
            metric_dict['95hd'].append(0)
        else:
            single_metric = calculate_metric_percase(prediction, label)
            metric_dict['name'].append(case_name)
            metric_dict['dice'].append(single_metric[0])
            metric_dict['jaccard'].append(single_metric[1])
            metric_dict['asd'].append(single_metric[2])
            metric_dict['95hd'].append(single_metric[3])
            # print(metric_dict)


        total_metric += np.asarray(single_metric)

        if save_result:
            test_save_path_temp = test_save_path
            if not os.path.exists(test_save_path_temp):
                os.makedirs(test_save_path_temp)
            sitk.WriteImage(sitk.GetImageFromArray(prediction.astype(np.float32)), test_save_path_temp + '/' + case_name + "_pred.nii.gz")
            # sitk.WriteImage(sitk.GetImageFromArray(image[:].astype(np.float32)), test_save_path_temp + '/' +  id + "_img.nii.gz")
            # sitk.WriteImage(sitk.GetImageFromArray(label[:].astype(np.float32)), test_save_path_temp + '/' + id + "_gt.nii.gz")
    avg_metric = total_metric / len(image_list)
    metric_csv = pd.DataFrame(metric_dict)
    metric_csv.to_csv(test_save_path + '/metric.csv', index=False)
    print('average metric is {}'.format(avg_metric))

    return avg_metric


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=10):
    w, h, d = image.shape
    thr = 0.5
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
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    # label_map = np.zeros(image.shape)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                print(f'#############    {int((x*sy*sz + y*sz + z) / (sx*sy*sz) * 100)}%', end='\r')
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                out = net(test_patch)
                out = out.cpu().data.numpy()
                out = out[0,:,:,:,:]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += out
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += 1

    # # modify
    # test_split = np.zeros([sx*sy*sz, 1, patch_size[0], patch_size[1], patch_size[2]]).astype(np.float32)
    # for x in range(0, sx):
    #     xs = min(stride_xy*x, ww-patch_size[0])
    #     for y in range(0, sy):
    #         ys = min(stride_xy * y,hh-patch_size[1])
    #         for z in range(0, sz):
    #             zs = min(stride_z * z, dd - patch_size[2])
    #             test_patch = image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
    #             test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(np.float32)
    #             test_split[x*sy*sz + y*sz + z, ...] = test_patch
    #
    # bs = 4
    # out = np.zeros_like(test_split)
    # out = np.repeat(out, 10, axis=1)
    # for i in range(test_split.shape[0] // bs):
    #     print(f'##########{int(i / (test_split.shape[0] // bs) * 100)}%', end='\r')
    #     with torch.no_grad():
    #         test_tmp = test_split[i*bs:min(test_split.shape[0], (i+1)*bs), ...]
    #         test_tmp = torch.from_numpy(test_tmp).cuda()
    #         # print(test_tmp.shape)
    #         # import pdb;pdb.set_trace()
    #         out_tmp = net(test_tmp)
    #         out_tmp = F.softmax(out_tmp, dim=1)
    #         out_tmp = out_tmp.cpu().data.numpy()
    #         out[i*bs:min(test_split.shape[0], (i+1)*bs), ...] = out_tmp
    #
    # for x in range(0, sx):
    #     xs = min(stride_xy*x, ww-patch_size[0])
    #     for y in range(0, sy):
    #         ys = min(stride_xy * y,hh-patch_size[1])
    #         for z in range(0, sz):
    #             zs = min(stride_z * z, dd-patch_size[2])
    #             score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] += out[x*sy*sz + y*sz + z, ...]
    #             cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] += 1

    # import pdb;pdb.set_trace()
    score_map = score_map/np.expand_dims(cnt,axis=0)
    # label_map[score_map > thr] = 1
    label_map = np.argmax(score_map, axis = 0)
    print()
    print(np.max(label_map))

    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
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
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, jc, hd, asd


if __name__ == '__main__':
    metric = test_calculate_metric()
    # print(metric)
