import torch
import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import pandas as pd
from NetworkTrainer.utils.util import AverageMeterArray
from NetworkTrainer.networks.unet import UNet3D, UNet3D_ds
from NetworkTrainer.utils.test_util import calculate_metric_percase
from NetworkTrainer.dataloaders.dataload import get_imglist
from NetworkTrainer.utils.post_process import *


class NetworkInfer:
    def __init__(self, opt):
        self.opt = opt

    def set_GPU_device(self):
       os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in self.opt.test['gpus'])
 
    def set_network(self):
        self.net = UNet3D(num_classes=self.opt.model['num_class'], input_channels=self.opt.model['in_c'], act='relu', norm=self.opt.train['norm'])
        if self.opt.train['deeps']:
            self.net = UNet3D_ds(num_classes=self.opt.model['num_class'], input_channels=self.opt.model['in_c'], act='relu', norm=self.opt.train['norm'])
             
        self.net = torch.nn.DataParallel(self.net)
        self.net = self.net.cuda()
        print(f"=> loading trained model in {self.opt.test['model_path']}")
        checkpoint = torch.load(self.opt.test['model_path'])
        self.net.load_state_dict(checkpoint['state_dict'])
        self.net.eval()
    
    def set_dataloader(self):
        image_list = get_imglist(self.opt.root_dir, self.opt.fold, 'test')
        self.image_list = [os.path.join(self.opt.root_dir, img) for img in image_list]
    
    def test_single_case(self, image):
        w, h, d = image.shape
        tta = TTA(if_flip=self.opt.test['flip'], if_rot=self.opt.test['rotate'])
        patch_size = self.opt.model['input_size']
        stride_xy = patch_size[0]//2
        stride_z = patch_size[2]//2
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
        score_map = np.zeros((self.opt.model['num_class'], ) + image.shape).astype(np.float32)
        cnt = np.zeros(image.shape).astype(np.float32)

        for x in range(0, sx):
            xs = min(stride_xy*x, ww-patch_size[0])
            for y in range(0, sy):
                ys = min(stride_xy * y,hh-patch_size[1])
                for z in range(0, sz):
                    zs = min(stride_z * z, dd-patch_size[2])
                    test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                    # apply tta
                    test_patch_list = tta.img_list(test_patch)
                    y_list = []
                    for img in test_patch_list:
                        img = np.expand_dims(np.expand_dims(img,axis=0),axis=0).astype(np.float32)
                        img = torch.from_numpy(img).cuda()
                        if not self.opt.train['deeps']:
                            y = self.net(img)
                        else:
                            y = self.net(img)[0]
                        y = F.softmax(y, dim=1)
                        y = y.cpu().detach().numpy()
                        y = np.squeeze(y)
                        y_list.append(y)
                    y_list = tta.img_list_inverse(y_list)
                    y = np.mean(y_list, axis=0)
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

    def post_process(self, pred):
        if self.opt.post['abl']:
            pred = abl(pred, for_which_classes=[1,])
        if self.opt.post['rsa']:
            pred = rsa(pred, for_which_classes=[1,2], minimum_valid_object_size={1: 1000, 2: 80})
        return pred

    def run(self):
        if self.opt.model['num_class'] == 2:
            metric_names = ['recall1', 'precision1', 'dice1', 'miou1']
        else:
            metric_names = ['recall1', 'precision1', 'dice1', 'miou1', 'recall2', 'precision2', 'dice2', 'miou2']
        total_metric = AverageMeterArray(len(metric_names))
        if self.opt.test['save_flag']:
            if not os.path.exists(self.opt.test['save_dir'] + '/img'):
                os.makedirs(self.opt.test['save_dir'] + '/img')
        for image_path in tqdm(self.image_list):
            case_name = os.path.basename(image_path)
            image = np.load(image_path+'_image.npy')
            label = np.load(image_path+'_label.npy')
            image = np.squeeze(image)
            label = np.squeeze(label)
            prediction, score_map = self.test_single_case(image)
            prediction = self.post_process(prediction)

            single_metric = calculate_metric_percase(prediction, label)
            total_metric.update([single_metric[metric_name] for metric_name in metric_names])

            if self.opt.test['save_flag']:
                np.save(os.path.join(self.opt.test['save_dir'], 'img', case_name+'_pred.npy'), prediction)
                np.save(os.path.join(self.opt.test['save_dir'], 'img', case_name+'_prob.npy'), score_map)
                # np.save(os.path.join(self.opt.test['save_dir'], 'img', case_name+'_img.npy'), image)
            print(case_name, single_metric)
        result_avg = [[total_metric.avg[i]*100 for i in range(len(metric_names))]]
        result_avg = pd.DataFrame(result_avg, columns=metric_names)
        result_avg.to_csv(os.path.join(self.opt.test['save_dir'], 'test_results.csv'), index=False)
