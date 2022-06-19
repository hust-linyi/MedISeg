
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
from models.modelU import ResUNet34
# from model_UNet import UNet
# from DenseUnet import UNet
from models.model_UNet import UNet
import utils.utils as utils
from utils.accuracy import compute_metrics
import time
import imageio
from options import Options
from utils.my_transforms import get_transforms
from rich import print
from tqdm import tqdm
from utils.dataset import DataFolder
from torch.utils.data import DataLoader
from scipy import ndimage
from multiprocessing import Pool
import cv2


def main():
    opt = Options(isTrain=False)
    opt.parse()
    opt.save_options()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.test['gpus'])

    save_dir = opt.test['save_dir']
    model_path = opt.test['model_path']
    save_flag = opt.test['save_flag']
    # img_dir = opt.test['img_dir']
    # label_dir = opt.test['label_dir']
    # img_path = os.path.join(opt.root_dir, 'test', 'data_after_stain_norm_ref1.npy')
    # gt_path = os.path.join(opt.root_dir, 'test', 'gt.npy')
    # img_path = os.path.join(opt.root_dir, 'data_after_stain_norm_ref1.npy')
    # gt_path = os.path.join(opt.root_dir, 'gt.npy')
    # bnd_path = os.path.join(opt.root_dir, 'bnd.npy')

    test_set = DataFolder(root_dir=opt.root_dir, phase='test', data_transform=opt.transform['test'], fold=opt.fold)
    test_loader = DataLoader(test_set, batch_size=opt.test['batch_size'], shuffle=False, drop_last=False)

    # data transforms
    # test_transform = get_transforms(opt.transform['test'])
    # test_set = DataFolder(root_dir=opt.root_dir, phase='val', data_transform=test_transform)
    # test_loader = DataLoader(test_set, batch_size=opt.test['batch_size'], shuffle=False, drop_last=False)
    
    if 'res' in opt.model['name']:
        model = ResUNet34(net=opt.model['name'], seg_classes=2, colour_classes=3)
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
    metric_names = ['acc', 'p_F1', 'p_recall', 'p_precision', 'dice', 'miou']
    test_results = dict()
    all_result = utils.AverageMeter(len(metric_names))

    for i, (input, gt, name) in enumerate(tqdm(test_loader)):
        input = input.cuda()

        outputs = model(input)
        output = outputs[0]
        pred = output.data.max(1)[1].cpu().numpy()

        for j in range(pred.shape[0]):
            metrics = compute_metrics(pred[j], gt[j], metric_names)
            all_result.update([metrics[metric_name] for metric_name in metric_names])
            if save_flag:
                imageio.imwrite(os.path.join(save_dir, 'img', f'{name[j]}_pred.png'), (pred[j] * 255).astype(np.uint8))
                imageio.imwrite(os.path.join(save_dir, 'img', f'{name[j]}_gt.png'), (gt[j].numpy() * 255).astype(np.uint8))

    for i in range(len(metric_names)):
        print(f"{metric_names[i]}: {all_result.avg[i]:.4f}", end='\t')

    header = metric_names
    utils.save_results(header, all_result.avg, test_results, '{:s}/test_results_epoch_{:d}_AJI_{:.4f}.txt'.format(save_dir, checkpoint['epoch'], all_result.avg[5]))


if __name__ == '__main__':
    main()
