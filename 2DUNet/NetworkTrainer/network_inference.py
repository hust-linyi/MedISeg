import torch
import os
import imageio
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import albumentations as A

from NetworkTrainer.networks.unet import UNet
from NetworkTrainer.networks.resunet import ResUNet, ResUNet_ds
from NetworkTrainer.networks.denseunet import DenseUNet
from NetworkTrainer.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from NetworkTrainer.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

from NetworkTrainer.dataloaders.dataload import DataFolder
from NetworkTrainer.utils.util import AverageMeterArray
from NetworkTrainer.utils.accuracy import compute_metrics
from NetworkTrainer.utils.post_process import *


class NetworkInference:
    def __init__(self, opt):
        self.opt = opt

    def set_GPU_device(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in self.opt.test['gpus'])

    def set_network(self):
        if 'res' in self.opt.model['name']:
            self.net = ResUNet(net=self.opt.model['name'], seg_classes=2, colour_classes=3, pretrained=self.opt.model['pretrained'])
            if self.opt.train['deeps']:
                self.net = ResUNet_ds(net=self.opt.model['name'], seg_classes=2, colour_classes=3, pretrained=self.opt.model['pretrained'])
        elif 'dense' in self.opt.model['name']:
            self.net = DenseUNet(net=self.opt.model['name'], seg_classes=2)
        elif 'trans' in self.opt.model['name']:
            config_vit = CONFIGS_ViT_seg[self.opt.model['name']]
            config_vit.n_classes = 2
            config_vit.n_skip = 4
            if self.opt.model['name'].find('R50') != -1:
                config_vit.patches.grid = (int(self.opt.model['input_size'][0] / 16), int(self.opt.model['input_size'][1] / 16))
            self.net = ViT_seg(config_vit, img_size=self.opt.model['input_size'][0], num_classes=config_vit.n_classes).cuda()
        else:
            self.net = UNet(3, 2, 2)
        self.net = torch.nn.DataParallel(self.net)
        self.net = self.net.cuda()

        # ----- load trained model ----- #
        print(f"=> loading trained model in {self.opt.test['model_path']}")
        checkpoint = torch.load(self.opt.test['model_path'])
        self.net.load_state_dict(checkpoint['state_dict'])
        print("=> loaded model at epoch {}".format(checkpoint['epoch']))
        self.net = self.net.module
        self.net.eval()
        
    def set_dataloader(self):
        test_set = DataFolder(root_dir=self.opt.root_dir, phase='test', data_transform=A.Compose(self.opt.transform['test']), fold=self.opt.fold)
        self.test_loader = DataLoader(test_set, batch_size=self.opt.test['batch_size'], shuffle=False, drop_last=False)

    def set_save_dir(self):
        if self.opt.test['save_flag']:
            if not os.path.exists(os.path.join(self.opt.test['save_dir'], 'img')):
                os.mkdir(os.path.join(self.opt.test['save_dir'], 'img'))
    
    def post_process(self, pred):
        if self.opt.post['abl']:
            pred = abl(pred, for_which_classes=[1])
        if self.opt.post['rsa']:
            # pred = rsa(pred, for_which_classes=[1], minimum_valid_object_size={1: 120})
            pred = rsa(pred, for_which_classes=[1], minimum_valid_object_size={1: 10})
        return pred

    def run(self):
        metric_names = ['p_recall', 'p_precision', 'p_F1', 'miou']
        all_result = AverageMeterArray(len(metric_names))
        for i, data in enumerate(tqdm(self.test_loader)):
            input, gt, name = data['image'].cuda(), data['label'], data['name']
            tta = TTA_2d(flip=self.opt.test['flip'], rotate=self.opt.test['rotate'])
            input_list = tta.img_list(input)
            y_list = []
            for x in input_list:
                x = torch.from_numpy(x.copy()).cuda()
                if not self.opt.train['deeps']:
                    y = self.net(x)
                else:
                    y = self.net(x)[0]
                y = torch.nn.Softmax(dim=1)(y)[:, 1]
                y = y.cpu().detach().numpy()
                y_list.append(y)
            y_list = tta.img_list_inverse(y_list)
            output = np.mean(y_list, axis=0)
            pred = (output > 0.5).astype(np.uint8)

            for j in range(pred.shape[0]):
                pred[j] = self.post_process(pred[j])
                metrics = compute_metrics(pred[j], gt[j], metric_names)
                if metrics[metric_names[0]] == -1:
                    continue
                all_result.update([metrics[metric_name] for metric_name in metric_names])
                if self.opt.test['save_flag']:
                    imageio.imwrite(os.path.join(self.opt.test['save_dir'], 'img', f'{name[j]}_pred.png'), (pred[j] * 255).astype(np.uint8))
                    imageio.imwrite(os.path.join(self.opt.test['save_dir'], 'img', f'{name[j]}_gt.png'), (gt[j].numpy() * 255).astype(np.uint8))
                    np.save(os.path.join(self.opt.test['save_dir'], 'img', f'{name[j]}_prob.npy'), output[j])

        for i in range(len(metric_names)):
            print(f"{metric_names[i]}: {all_result.avg[i]:.4f}", end='\t')

        result_avg = [[all_result.avg[i]*100 for i in range(len(metric_names))]]
        result_avg = pd.DataFrame(result_avg, columns=metric_names)
        result_avg.to_csv(os.path.join(self.opt.test['save_dir'], 'test_results.csv'), index=False)