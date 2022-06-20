# -*- encoding: utf-8 -*-
import time
import torch
import torch.nn as nn
from torch import optim
import os
from utils.util import save_bestcheckpoint, save_checkpoint, setup_logging 
import random
import numpy as np
from networks.unet import UNet3D
import os
from tqdm import tqdm
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from networks.unet import UNet3D
from utils.util import AverageMeter
from dataloaders.data_kit import DataFolder
from rich.logging import RichHandler
from utils.util import save_bestcheckpoint, save_checkpoint, setup_logging 
import logging


class TrainerSetting:
    def __init__(self):
        self.project_name = None
        # Path for saving model and training log
        self.output_dir = None

        # Generally only use one of them
        self.max_iter = 99999999
        self.max_epoch = 99999999

        # Default not use this,
        # because the models of "best_train_loss", "best_val_evaluation_index", "latest" have been saved.
        self.save_per_epoch = 99999999
        self.eps_train_loss = 0.01

        self.network = None
        self.device = None
        self.list_GPU_ids = None

        self.train_loader = None
        self.val_loader = None

        self.optimizer = None
        self.lr_scheduler = None
        self.lr_scheduler_type = None

        # Default update learning rate after each epoch
        self.lr_scheduler_update_on_iter = False

        self.loss_function = None

        # If do online evaluation during validation
        self.online_evaluation_function_val = None


class TrainerLog:
    def __init__(self):
        self.iter = -1
        self.epoch = -1

        # Moving average loss, loss is the smaller the better
        self.moving_train_loss = None
        # Average train loss of a epoch
        self.average_train_loss = 99999999.
        self.best_average_train_loss = 99999999.
        # Evaluation index is the higher the better
        self.average_val_index = -99999999.
        self.best_average_val_index = -99999999.

        # Record changes in training loss
        self.list_average_train_loss_associate_iter = []
        # Record changes in validation evaluation index
        self.list_average_val_index_associate_iter = []
        # Record changes in learning rate
        self.list_lr_associate_iter = []

        # Save status of the trainer, eg. best_train_loss, latest, best_val_evaluation_index
        self.save_status = []


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
        train_set = DataFolder(root_dir=self.opt.root_dir, phase='train', fold=self.opt.fold, data_transform=self.opt.transform['train'])
        self.train_loader = DataLoader(train_set, batch_size=self.opt.train['batch_size'], shuffle=True, num_workers=self.opt.train['workers'])
        val_set = DataFolder(root_dir=self.opt.root_dir, phase='val', data_transform=self.opt.transform['val'], fold=self.opt.fold)
        self.val_loader = DataLoader(val_set, batch_size=self.opt.train['batch_size'], shuffle=False, drop_last=False, num_workers=self.opt.train['workers'])


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

