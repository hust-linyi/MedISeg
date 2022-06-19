# from genericpath import exists
import torch
# from torch import tensor
# from torch.autograd import Variable
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.utils.data
import os
import shutil
import numpy as np
import logging
# from tensorboardX import SummaryWriter
import random
from models.modelU import ResUNet34, ResUNet
# from DenseUnet import UNet
from models.model_UNet import UNet
import utils.utils as utils
# from dataset_monuseg_brp_ori_mix import DataFolder
from utils.dataset import DataFolder
from utils.my_transforms import get_transforms
from options import Options
# from pytorchtools import EarlyStopping
from rich.logging import RichHandler
from rich import print
from tqdm import tqdm
# from PIL import Image
# import imageio
import wandb
from utils.loss import dice_loss, smooth_truncated_loss




def main():
    global opt, num_iter, tb_writer, logger, logger_results
    opt = Options(isTrain=True)
    opt.parse()
    opt.save_options()

    # !wandb login
    wandb.init(project="isic2018", dir='{:s}'.format(opt.train['save_dir']))
    wandb.watch_called = False  # Re-run the model without restarting the runtime, unnecessary after our next release
    # wandb = None

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.train['gpus'])

    # set up logger
    logger, logger_results = setup_logging(opt)
    num = 2022
    random.seed(num)
    os.environ['PYTHONHASHSEED'] = str(num)
    np.random.seed(num)
    torch.manual_seed(num)
    torch.cuda.manual_seed(num)
    torch.cuda.manual_seed_all(num)

    # ----- create model ----- #
    if 'res' in opt.model['name']:
        model = ResUNet(net=opt.model['name'], seg_classes=2, colour_classes=3, pretrained=opt.model['pretrained'])
    else:
        model = UNet(3, 2, 2)
    model = nn.DataParallel(model)
    model = model.cuda()

    cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # ----- define optimizer ----- #
    optimizer = torch.optim.Adam(model.parameters(), opt.train['lr'], betas=(0.9, 0.99), weight_decay=opt.train['weight_decay'])
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.train['train_epochs'])
    # ----- define criterion ----- #
    # criterion = torch.nn.NLLLoss(ignore_index=2).cuda()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=7).cuda()

    # ----- load data ----- #
    # data_transforms = {'train': get_transforms(opt.transform['train']),
    #                    'val': get_transforms(opt.transform['val']),
    #                    'test': get_transforms(opt.transform['test'])}
    # data_transforms = {'train': opt.transform['train'],
    #                    'val': opt.transform['val'],
    #                    'test': opt.transform['test']}

    train_set = DataFolder(root_dir=opt.root_dir, phase='train', fold=opt.fold, data_transform=opt.transform['train'])
    train_loader = DataLoader(train_set, batch_size=opt.train['batch_size'], shuffle=True, num_workers=opt.train['workers'])
    val_set = DataFolder(root_dir=opt.root_dir, phase='test', data_transform=opt.transform['val'], fold=opt.fold)
    val_loader = DataLoader(val_set, batch_size=opt.train['batch_size'], shuffle=False, drop_last=False, num_workers=opt.train['workers'])

    # ----- training and validation ----- #
    num_epoch = opt.train['train_epochs']
    # print training parameters
    logger.info("=> Initial learning rate: {:g}".format(opt.train['lr']))
    logger.info("=> Batch size: {:d}".format(opt.train['batch_size']))
    logger.info("=> Number of training iterations: {:d} * {:d}".format(num_epoch, int(len(train_loader))))
    logger.info("=> Training epochs: {:d}".format(opt.train['train_epochs']))
    min_loss = 100
    dataprocess = tqdm(range(opt.train['start_epoch'], num_epoch))
    for epoch in dataprocess:
        train_results = train(train_loader, model, optimizer, criterion, epoch, opt.task)
        state = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}

        # save the training results to txt files
        logger_results.info('{:d}\t{:.4f}'.format(epoch+1, train_results[0]))
        scheduler.step()
        # val_loss = val(val_loader, model, criterion)
        val_loss = 0
        if val_loss < min_loss:
            dataprocess.set_description(f"val_loss: {val_loss:.4f}, min_loss: {min_loss:.4f}")
            # print('val_loss:', f'{val_loss:.4f}', 'min_loss:', f'{min_loss:.4f}')
            min_loss = val_loss
            save_bestcheckpoint(state, opt.train['save_dir'])
        if (epoch + 1) < (num_epoch - 4):
            cp_flag = (epoch + 1) % opt.train['checkpoint_freq'] == 0
        else:
            cp_flag = True
        save_checkpoint(state, epoch, opt.train['save_dir'], cp_flag)

    save_wandb(opt)
    for i in list(logger.handlers):
        logger.removeHandler(i)
        i.flush()
        i.close()
    for i in list(logger_results.handlers):
        logger_results.removeHandler(i)
        i.flush()
        i.close()


def train(train_loader, model, optimizer, criterion, epoch, mix_loss='baseline', thr_epoch=30, thr_conf=0.05):
    # list to store the average loss for this epoch
    results = utils.AverageMeter()
    # switch to train mode
    model.train()
    wb_img, wb_label, wb_pred = [], [], []
    torch.autograd.set_detect_anomaly(True)
    for i, (input, gt, _) in enumerate(train_loader):
        input, gt = input.cuda(), gt.cuda().long().squeeze(1)

        output = model(input)
        loss_ce = criterion(output, gt)
        prob = F.softmax(output, dim=1)[:, 1, :, :]
        loss_dice = dice_loss(prob, gt)
        loss = loss_ce + loss_dice

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        results.update([loss.item()], input.size(0))

        if len(wb_img) < 5:
            pred = torch.argmax(output, axis=1)
            _wb_img = (input[0].cpu().detach().numpy() * 255).transpose(1,2,0).astype(np.uint8)
            _wb_pred = (pred[0].cpu().detach().numpy() * 255).astype(np.uint8)
            _wb_gt = (gt[0].cpu().detach().numpy() * 255).astype(np.uint8)

            _wb_pred_tmp = _wb_img * 0.7
            _wb_label_tmp = _wb_img * 0.7
            _wb_pred_tmp[..., 2] = _wb_pred
            _wb_label_tmp[..., 2] = _wb_gt

            wb_img.append(_wb_img)
            wb_label.append(_wb_pred_tmp)
            wb_pred.append(_wb_label_tmp)

    wandb.log({
        'train_loss': results.avg[0]
    })    
    wandb.log({
        "train-image": [wandb.Image(e) for e in wb_img],
        "train-pred": [wandb.Image(e) for e in wb_pred],
        "train-label": [wandb.Image(e) for e in wb_label],
        })
    return results.avg


def val(val_loader, model, criterion):
    model.eval()
    results = 0
    wb_img, wb_label, wb_pred = [], [], []
    for i, sample in enumerate(val_loader):
        input, gt, _ = sample
        input = input.cuda()
        gt = gt.cuda().long().squeeze(1)

        # compute output
        output = model(input)
        loss = criterion(output, gt)
        results += loss.item()
        pred = torch.argmax(output, axis=1)
        if len(wb_img) < 5:
            _wb_img = (input[0].cpu().detach().numpy() * 255).transpose(2,1,0).astype(np.uint8)
            _wb_pred = (pred[0].cpu().detach().numpy() * 255).transpose(1,0).astype(np.uint8)
            _wb_label = (gt[0].cpu().detach().numpy() * 255).transpose(1,0).astype(np.uint8)
            wb_img.append(_wb_img)
            wb_label.append(_wb_label)
            wb_pred.append(_wb_pred)

    val_loss = results / (opt.train['batch_size'] * len(val_loader))
    wandb.log({
        'test_loss': val_loss, 
    })
    wandb.log({
        "image": [wandb.Image(e) for e in wb_img],
        "pred": [wandb.Image(e) for e in wb_pred],
        "label": [wandb.Image(e) for e in wb_label],
        })
    return val_loss
 

def save_checkpoint(state, epoch, save_dir, cp_flag):
    cp_dir = '{:s}/checkpoints'.format(save_dir)
    if not os.path.exists(cp_dir):
        os.mkdir(cp_dir)
    filename = '{:s}/checkpoint_999.pth.tar'.format(cp_dir)
    torch.save(state, filename)
    if cp_flag:
        shutil.copyfile(filename, '{:s}/checkpoint_{:d}.pth.tar'.format(cp_dir, epoch+1))


def save_bestcheckpoint(state, save_dir):
    cp_dir = '{:s}/checkpoints'.format(save_dir)
    torch.save(state, '{:s}/checkpoint_0.pth.tar'.format(cp_dir))


def setup_logging(opt):
    mode = 'a' if opt.train['checkpoint'] else 'w'

    # create logger for training information
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.DEBUG)
    # create console handler and file handler
    # console_handler = logging.StreamHandler()
    console_handler = RichHandler(show_level=False, show_time=False, show_path=False)
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler('{:s}/train_log.txt'.format(opt.train['save_dir']), mode=mode)
    file_handler.setLevel(logging.DEBUG)
    # create formatter
    # formatter = logging.Formatter('%(asctime)s\t%(message)s', datefmt='%m-%d %I:%M')
    formatter = logging.Formatter('%(message)s')
    # add formatter to handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    # add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # create logger for epoch results
    logger_results = logging.getLogger('results')
    logger_results.setLevel(logging.DEBUG)
    file_handler2 = logging.FileHandler('{:s}/epoch_results.txt'.format(opt.train['save_dir']), mode=mode)
    file_handler2.setFormatter(logging.Formatter('%(message)s'))
    logger_results.addHandler(file_handler2)

    logger.info('***** Training starts *****')
    logger.info('save directory: {:s}'.format(opt.train['save_dir']))
    if mode == 'w':
        logger_results.info('epoch\ttrain_loss\ttrain_loss_vor\ttrain_loss_cluster\ttrain_loss_repel')

    return logger, logger_results


def save_wandb(opt):
    config = wandb.config  # Initialize config
    config.batch_size = opt.train['batch_size']  # input batch size for training (default:64)
    config.epochs = opt.train['train_epochs']  # number of epochs to train(default:10)
    config.lr = opt.train['lr']  # learning rate(default:0.01)
    config.input_size = opt.train['input_size']
    config.workers = opt.train['workers']
    config.gpu = opt.train['gpus']

if __name__ == '__main__':
    main()
