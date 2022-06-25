import torch
import os
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
from NetworkTrainer.utils.util import AverageMeterArray
from sklearn.metrics import recall_score, precision_score, f1_score, jaccard_score
from NetworkTrainer.networks.unet import UNet3D
from NetworkTrainer.utils.test_util import test_all_case
from NetworkTrainer.dataloaders.data_kit import get_imglist


def test_calculate_metric(opt):
    net = UNet3D(num_classes=3, input_channels=1, act='relu', norm=opt.train['norm'])
    net = torch.nn.DataParallel(net)
    net = net.cuda()

    print(f"=> loading trained model in {opt.test['model_path']}")
    checkpoint = torch.load(opt.test['model_path'])
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    image_list = get_imglist(opt.root_dir, opt.fold, 'test')
    image_list = [os.path.join(opt.root_dir, img) for img in image_list]

    test_all_case(net, image_list, num_classes=3,
                               patch_size=opt.model['input_size'], stride_xy=opt.model['input_size'][0]//2, stride_z=opt.model['input_size'][0]//2,
                               save_result=opt.test['save_flag'], test_save_path=opt.test['save_dir'])


class NetworkInfer:
    def __init__(self, opt):
        self.opt = opt

    def set_GPU_device(self):
       os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in self.opt.test['gpus'])
 
    def set_network(self):
        self.net = UNet3D(num_classes=3, input_channels=1, act='relu', norm=self.opt.train['norm'])
        self.net = torch.nn.DataParallel(self.net)
        self.net = self.net.cuda()
        print(f"=> loading trained model in {self.opt.test['model_path']}")
        checkpoint = torch.load(self.opt.test['model_path'])
        self.net.load_state_dict(checkpoint['state_dict'])
        self.net.eval()
    
    def set_dataloader(self):
        image_list = get_imglist(self.opt.root_dir, self.opt.fold, 'test')
        self.image_list = [os.path.join(self.opt.root_dir, img) for img in image_list]
        
    def run(self):
        metric_names = ['recall1', 'precision1', 'dice1', 'miou1', 'recall2', 'precision2', 'dice2', 'miou2']
        total_metric = AverageMeterArray(len(metric_names))
        if self.opt.test['save_flag']:
            if not os.path.exists(self.opt.test['save_dir'] + '/img'):
                os.makedirs(self.opt.test['save_dir'] + '/img')
        for image_path in tqdm(image_list):
            case_name = os.path.basename(image_path)
            image = np.load(image_path+'_image.npy')
            label = np.load(image_path+'_label.npy')
            image = np.squeeze(image)
            label = np.squeeze(label)
            prediction, score_map = test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)

            single_metric = calculate_metric_percase(prediction, label)
            total_metric.update([single_metric[metric_name] for metric_name in metric_names])

            if opt.test['save_flag']:
                np.save(os.path.join(self.opt.test['save_dir'], 'img', case_name+'_pred.npy'), prediction)
                # np.save(os.path.join(self.opt.test['save_dir'], 'img', case_name+'_gt.npy'), label)
                # np.save(os.path.join(self.opt.test['save_dir'], 'img', case_name+'_prob.npy'), score_map)
                # np.save(os.path.join(self.opt.test['save_dir'], 'img', case_name+'_img.npy'), image)
            print(case_name, single_metric)
        result_avg = [[total_metric.avg[i]*100 for i in range(len(metric_names))]]
        result_avg = pd.DataFrame(result_avg, columns=metric_names)
        result_avg.to_csv(os.path.join(self.opt.test['save_dir'], 'test_results.csv'), index=False)
