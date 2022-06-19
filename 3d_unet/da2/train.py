import os
from tqdm import tqdm
import shutil
import logging
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from networks.unet import UNet3D
from utils.util import AverageMeter
from dataloaders.data_kit import DataFolder
from options import Options
from rich.logging import RichHandler


def val(net, valloader):
    net.eval()
    val_losses = AverageMeter()
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(valloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            outputs = net(volume_batch)

            val_loss = torch.nn.CrossEntropyLoss()(outputs, label_batch[:, 0, ...].long())
            val_losses.update(val_loss.item(), outputs.size(0))
    return val_losses.avg

def train(net, trainloader, optimizer, epoch):
    net.train()
    losses = AverageMeter()
    for i_batch, sampled_batch in enumerate(trainloader):
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
        outputs = net(volume_batch)
        loss_ce = torch.nn.CrossEntropyLoss()(outputs, label_batch[:, 0, ...].long())
        loss = loss_ce

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), outputs.size(0))
        print(f'debug: {volume_batch.sum()}')
        print(f'train epoch {epoch} batch {i_batch} loss {loss.item():.4f}')
    return losses.avg


def main():
    global opt
    opt = Options(isTrain=True)
    opt.parse()
    opt.save_options()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.train['gpus'])
    logger, logger_results = setup_logging(opt)

    num = opt.train['seed']
    random.seed(num)
    os.environ['PYTHONHASHSEED'] = str(num)
    np.random.seed(num)
    torch.manual_seed(num)
    torch.cuda.manual_seed(num)
    torch.cuda.manual_seed_all(num)

    net = UNet3D(num_classes=3, input_channels=1, act='relu', norm=opt.train['norm'])
    net = torch.nn.DataParallel(net)
    net = net.cuda()

    # optimizer = optim.Adam(net.parameters(), lr=opt.train['lr'], weight_decay=opt.train['weight_decay'])
    optimizer = optim.SGD(net.parameters(), lr=opt.train['lr'], momentum=0.9, weight_decay=opt.train['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    train_set = DataFolder(root_dir=opt.root_dir, phase='train', fold=opt.fold, data_transform=opt.transform['train'])
    train_loader = DataLoader(train_set, batch_size=opt.train['batch_size'], shuffle=True, num_workers=opt.train['workers'])
    val_set = DataFolder(root_dir=opt.root_dir, phase='val', data_transform=opt.transform['val'], fold=opt.fold)
    val_loader = DataLoader(val_set, batch_size=opt.train['batch_size'], shuffle=False, drop_last=False, num_workers=opt.train['workers'])

    num_epoch = opt.train['train_epochs']
    logger.info("=> Initial learning rate: {:g}".format(opt.train['lr']))
    logger.info("=> Batch size: {:d}".format(opt.train['batch_size']))
    logger.info("=> Number of training iterations: {:d} * {:d}".format(num_epoch, int(len(train_loader))))
    logger.info("=> Training epochs: {:d}".format(opt.train['train_epochs']))

    dataprocess = tqdm(range(opt.train['start_epoch'], num_epoch))
    best_val_loss = 100.0
    for epoch in dataprocess:
        state = {'epoch': epoch + 1, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}
        train_loss = train(net, train_loader, optimizer, epoch)
        val_loss = val(net, val_loader)
        scheduler.step(val_loss)
        logger_results.info('{:d}\t{:.4f}\t{:.4f}'.format(epoch+1, train_loss, val_loss))

        if val_loss<best_val_loss:
            best_val_loss = val_loss
            save_bestcheckpoint(state, opt.train['save_dir'])
            print(f'save best checkpoint at epoch {epoch}')
        if epoch % opt.train['checkpoint_freq'] == 0:
            save_checkpoint(state, epoch, opt.train['save_dir'], True)

    logging.info("training finished")


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


if __name__ == "__main__":
    main()