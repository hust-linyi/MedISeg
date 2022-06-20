import torch
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
import random
from models.modelU import ResUNet
from models.model_UNet import UNet
import utils.utils as utils
from utils.dataset import DataFolder
# from utils.my_transforms import get_transforms
from options import Options
from rich.logging import RichHandler
from rich import print
from tqdm import tqdm
from utils.loss import dice_loss


def main():
    global opt, num_iter, tb_writer, logger, logger_results
    opt = Options(isTrain=True)
    opt.parse()
    opt.save_options()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.train['gpus'])

    # set up logger
    logger, logger_results = setup_logging(opt)
    num = opt.train['seed']
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.train['train_epochs'])
    # ----- define criterion ----- #
    criterion = torch.nn.CrossEntropyLoss(ignore_index=7).cuda()

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
        val_loss = val(val_loader, model, dice_loss)
        if val_loss < min_loss:
            dataprocess.set_description(f"val_loss: {val_loss:.4f}, min_loss: {min_loss:.4f}")
            min_loss = val_loss
            save_bestcheckpoint(state, opt.train['save_dir'])
        if (epoch + 1) < (num_epoch - 4):
            cp_flag = (epoch + 1) % opt.train['checkpoint_freq'] == 0
        else:
            cp_flag = True
        save_checkpoint(state, epoch, opt.train['save_dir'], cp_flag)

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
    torch.autograd.set_detect_anomaly(True)
    for i, (input, gt, _) in enumerate(train_loader):
        input, gt = input.cuda(), gt.cuda().long().squeeze(1)

        output = model(input)
        loss_ce = criterion(output, gt)
        prob = F.softmax(output, dim=1)[:, 1, :, :]
        loss = loss_ce

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        results.update([loss.item()], input.size(0))
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
        prob = F.softmax(output, dim=1)[:, 1, :, :]
        loss = criterion(prob, gt)
        results += loss.item()
        pred = torch.argmax(output, axis=1)
    val_loss = results / (opt.train['batch_size'] * len(val_loader))
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


if __name__ == '__main__':
    main()