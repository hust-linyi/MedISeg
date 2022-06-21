# -*- encoding: utf-8 -*-
import random
import os
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from NetworkTrainer.networks.unet import UNet3D
from NetworkTrainer.dataloaders.data_kit import DataFolder
from NetworkTrainer.utils.util import save_bestcheckpoint, save_checkpoint, setup_logging, AverageMeter
import logging
from torchvision import transforms


class NetworkTrainer:
    def __init__(self, opt):
        self.opt = opt

    def set_GPU_device(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in self.opt.train['gpus'])
    
    def set_logging(self):
        self.logger, self.logger_results = setup_logging(self.opt)
    
    def set_randomseed(self):
        num = self.opt.train['seed']
        random.seed(num)
        os.environ['PYTHONHASHSEED'] = str(num)
        np.random.seed(num)
        torch.manual_seed(num)
        torch.cuda.manual_seed(num)
        torch.cuda.manual_seed_all(num)
    
    def set_network(self):
        self.net = UNet3D(num_classes=3, input_channels=1, act='relu', norm=self.opt.train['norm'])
        self.net = torch.nn.DataParallel(self.net)
        self.net = self.net.cuda()

    def set_optimizer(self):
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.opt.train['lr'], momentum=0.9, weight_decay=self.opt.train['weight_decay'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

    def set_dataloader(self):
        # self.train_set = DataFolder(root_dir=self.opt.root_dir, phase='train', fold=self.opt.fold, data_transform=self.opt.transform['train'])
        # self.val_set = DataFolder(root_dir=self.opt.root_dir, phase='val', data_transform=self.opt.transform['val'], fold=self.opt.fold)
        self.train_set = DataFolder(root_dir=self.opt.root_dir, phase='train', fold=self.opt.fold, data_transform=transforms.Compose(self.opt.transform['train']))
        self.val_set = DataFolder(root_dir=self.opt.root_dir, phase='val', data_transform=transforms.Compose(self.opt.transform['val']), fold=self.opt.fold)
        self.train_loader = DataLoader(self.train_set, batch_size=self.opt.train['batch_size'], shuffle=True, num_workers=self.opt.train['workers'])
        self.val_loader = DataLoader(self.val_set, batch_size=self.opt.train['batch_size'], shuffle=False, drop_last=False, num_workers=self.opt.train['workers'])


    def train(self, epoch):
        self.net.train()
        losses = AverageMeter()
        for i_batch, sampled_batch in enumerate(self.train_loader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            outputs = self.net(volume_batch)
            loss_ce = torch.nn.CrossEntropyLoss()(outputs, label_batch[:, 0, ...].long())
            loss = loss_ce

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.update(loss.item(), outputs.size(0))
            print(f'train epoch {epoch} batch {i_batch} loss {loss.item():.4f}')
        return losses.avg


    def val(self):
        self.net.eval()
        val_losses = AverageMeter()
        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(self.val_loader):
                volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
                outputs = self.net(volume_batch)

                val_loss = torch.nn.CrossEntropyLoss()(outputs, label_batch[:, 0, ...].long())
                val_losses.update(val_loss.item(), outputs.size(0))
        return val_losses.avg


    def run(self):
        num_epoch = self.opt.train['train_epochs']
        self.logger.info("=> Initial learning rate: {:g}".format(self.opt.train['lr']))
        self.logger.info("=> Batch size: {:d}".format(self.opt.train['batch_size']))
        self.logger.info("=> Number of training iterations: {:d} * {:d}".format(num_epoch, int(len(self.train_loader))))
        self.logger.info("=> Training epochs: {:d}".format(self.opt.train['train_epochs']))

        dataprocess = tqdm(range(self.opt.train['start_epoch'], num_epoch))
        best_val_loss = 100.0    
        for epoch in dataprocess:
            state = {'epoch': epoch + 1, 'state_dict': self.net.state_dict(), 'optimizer': self.optimizer.state_dict()}
            train_loss = self.train(epoch)
            val_loss = self.val()
            self.scheduler.step(val_loss)
            self.logger_results.info('{:d}\t{:.4f}\t{:.4f}'.format(epoch+1, train_loss, val_loss))

            if val_loss<best_val_loss:
                best_val_loss = val_loss
                save_bestcheckpoint(state, self.opt.train['save_dir'])

                print(f'save best checkpoint at epoch {epoch}')
            if epoch % self.opt.train['checkpoint_freq'] == 0:
                save_checkpoint(state, epoch, self.opt.train['save_dir'], True)

        logging.info("training finished")

