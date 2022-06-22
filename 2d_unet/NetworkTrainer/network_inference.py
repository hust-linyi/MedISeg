import torch
from NetworkTrainer.networks.unet import UNet
from NetworkTrainer.networks.resunet import ResUNet
import os
from dataloaders.dataset import DataFolder
from torch.utils.data import DataLoader
from utils.util import AverageMeterArray
from tqdm import tqdm
from utils.accuracy import compute_metrics
import imageio
import pandas as pd
import numpy as np


class NetworkInference:
    def __init__(self, opt):
        opt = opt

    def set_GPU_device(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in self.opt.train['gpus'])

    def set_network(self):
        if 'res' in self.opt.model['name']:
            self.model = ResUNet(net=self.model['name'], seg_classes=2, colour_classes=3, pretrained=self.opt.model['pretrained'])
        else:
            self.model = UNet(3, 2, 2)
        self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.cuda()
        # ----- load trained model ----- #
        print(f"=> loading trained model in {self.opt.test['model_path']}")
        checkpoint = torch.load(self.opt.test['model_path'])
        self.model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded model at epoch {}".format(checkpoint['epoch']))
        self.model = self.model.module
        self.model.eval()
        
    def set_dataloader(self):
        test_set = DataFolder(root_dir=self.opt.root_dir, phase='test', data_transform=self.opt.transform['test'], fold=self.opt.fold)
        self.test_loader = DataLoader(test_set, batch_size=self.opt.test['batch_size'], shuffle=False, drop_last=False)

    def set_save_dir(self):
        if self.opt.test['save_flag']:
            if not os.path.exists(os.path.join(self.opt.test['save_dir'], 'img')):
                os.mkdir(os.path.join(self.opt.test['save_dir'], 'img'))
    
    def run(self):
        metric_names = ['p_recall', 'p_precision', 'dice', 'miou']
        all_result = AverageMeterArray(len(metric_names))
        for i, (input, gt, name) in enumerate(tqdm(self.test_loader)):
            input = input.cuda()

            output = self.model(input)
            pred = output.data.max(1)[1].cpu().numpy()

            for j in range(pred.shape[0]):
                metrics = compute_metrics(pred[j], gt[j], metric_names)
                all_result.update([metrics[metric_name] for metric_name in metric_names])
                if self.opt.test['save_flag']:
                    imageio.imwrite(os.path.join(self.opt.test['save_dir'], 'img', f'{name[j]}_pred.png'), (pred[j] * 255).astype(np.uint8))
                    imageio.imwrite(os.path.join(self.opt.test['save_dir'], 'img', f'{name[j]}_gt.png'), (gt[j].numpy() * 255).astype(np.uint8))

        for i in range(len(metric_names)):
            print(f"{metric_names[i]}: {all_result.avg[i]:.4f}", end='\t')

        result_avg = [[all_result.avg[i]*100 for i in range(len(metric_names))]]
        result_avg = pd.DataFrame(result_avg, columns=metric_names)
        result_avg.to_csv(os.path.join(self.opt.test['save_dir'], 'test_results.csv'), index=False)