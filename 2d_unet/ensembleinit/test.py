
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import skimage.morphology as morph
import scipy.ndimage.morphology as ndi_morph
from skimage import measure
from scipy import misc
from models.modelU import ResUNet
# from model_UNet import UNet
# from DenseUnet import UNet
from models.model_UNet import UNet
import utils.utils as utils
from utils.accuracy import compute_metrics
import time
import imageio
from options import Options
from rich import print
from tqdm import tqdm
from utils.dataset import DataFolder
from torch.utils.data import DataLoader
from scipy import ndimage
from multiprocessing import Pool
import cv2
import pandas as pd


def main():
    opt = Options(isTrain=False)
    opt.parse()
    opt.save_options()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.test['gpus'])

    save_dir = opt.test['save_dir']
    model_path = opt.test['model_path']
    save_flag = opt.test['save_flag']

    test_set = DataFolder(root_dir=opt.root_dir, phase='test', data_transform=opt.transform['test'], fold=opt.fold)
    test_loader = DataLoader(test_set, batch_size=opt.test['batch_size'], shuffle=False, drop_last=False)

    if 'res' in opt.model['name']:
        model = ResUNet(net=opt.model['name'], seg_classes=2, colour_classes=3)
    else:
        model = UNet(3, 2, 2)    
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    cudnn.benchmark = True

    # ----- load trained model ----- #
    print(f"=> loading trained model in {model_path}")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    print("=> loaded model at epoch {}".format(checkpoint['epoch']))
    model = model.module

    # switch to evaluate mode
    model.eval()

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if save_flag:
        if not os.path.exists(os.path.join(save_dir, 'img')):
            os.mkdir(os.path.join(save_dir, 'img'))

    # metric_names = ['acc', 'p_F1', 'p_recall', 'p_precision', 'dice', 'aji']
    metric_names = ['p_recall', 'p_precision', 'dice', 'miou']
    test_results = dict()
    all_result = utils.AverageMeter(len(metric_names))

    for i, (input, gt, name) in enumerate(tqdm(test_loader)):
        input = input.cuda()

        output = model(input)
        pred = output.data.max(1)[1].cpu().numpy()
        prob = F.softmax(output, dim=1)[:, 1, :, :]

        for j in range(pred.shape[0]):
            metrics = compute_metrics(pred[j], gt[j], metric_names)
            all_result.update([metrics[metric_name] for metric_name in metric_names])
            if save_flag:
                imageio.imwrite(os.path.join(save_dir, 'img', f'{name[j]}_pred.png'), (pred[j] * 255).astype(np.uint8))
                imageio.imwrite(os.path.join(save_dir, 'img', f'{name[j]}_gt.png'), (gt[j].numpy() * 255).astype(np.uint8))
                np.save(os.path.join(save_dir, 'img', f'{name[j]}_prob.npy'), prob[j].detach().cpu().numpy())

    for i in range(len(metric_names)):
        print(f"{metric_names[i]}: {all_result.avg[i]:.4f}", end='\t')

    # header = metric_names
    # utils.save_results(header, all_result.avg, test_results, '{:s}/test_results_epoch_{:d}_AJI_{:.4f}.txt'.format(save_dir, checkpoint['epoch'], all_result.avg[5]))
    result_avg = [[all_result.avg[i]*100 for i in range(len(metric_names))]]
    result_avg = pd.DataFrame(result_avg, columns=metric_names)
    result_avg.to_csv(os.path.join(save_dir, 'test_results.csv'), index=False)



if __name__ == '__main__':
    main()
