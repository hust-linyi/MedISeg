from ctypes import resize
import os
import numpy as np
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Options:
    def __init__(self, isTrain):
        self.isTrain = isTrain
        self.model = dict()
        self.train = dict()
        self.test = dict()
        self.transform = dict()
        self.post = dict()

    def parse(self):
        """ Parse the options, replace the default value if there is a new input """
        parser = argparse.ArgumentParser(description='')
        parser.add_argument('--dataset', type=str, default='isic2018', help='dataset name')
        parser.add_argument('--task', type=str, default='baseline', help='baseline, cutout, cutmix, cowout, cowmix, insmix')
        parser.add_argument('--fold', type=int, default=0, help='0-4, five fold cross validation')
        parser.add_argument('--name', type=str, default='res50', help='res34, res50, res101, res152')
        parser.add_argument('--pretrained', type=bool, default=False, help='True or False')
        parser.add_argument('--fix-params', type=bool, default=False, help='True or False')
        parser.add_argument('--in-c', type=int, default=3, help='input channel')
        parser.add_argument('--train-input-size', type=list, default=[256,192], help='input size of the image')
        parser.add_argument('--train-train-epochs', type=int, default=200, help='number of training epochs')
        parser.add_argument('--train-batch-size', type=int, default=32, help='batch size')
        parser.add_argument('--train-checkpoint-freq', type=int, default=20, help='epoch to save checkpoints')
        parser.add_argument('--train-lr', type=float, default=3e-4, help='initial learning rate')
        parser.add_argument('--train-weight-decay', type=float, default=1e-5, help='weight decay')
        parser.add_argument('--train-log-interval', type=int, default=37, help='iterations to print training results')
        parser.add_argument('--train-workers', type=int, default=16, help='number of workers to load images')
        parser.add_argument('--train-gpus', type=list, default=[0, ], help='select gpu devices')
        parser.add_argument('--train-start-epoch', type=int, default=0, help='start epoch')
        parser.add_argument('--train-checkpoint', type=str, default='', help='checkpoint')
        parser.add_argument('--train-seed', type=int, default=2022, help='test epoch')
        parser.add_argument('--test-test-epoch', type=int, default=0, help='test epoch')
        parser.add_argument('--test-gpus', type=list, default=[0, ], help='select gpu devices')
        parser.add_argument('--test-save-flag', type=bool, default=True, help='True or False')
        parser.add_argument('--test-patch-size', type=int, default=224, help='input size of the image')
        parser.add_argument('--test-overlap', type=int, default=80, help='overlap')
        parser.add_argument('--test-batch-size', type=int, default=4, help='batch size')
        args = parser.parse_args()

        self.dataset = args.dataset
        self.task = args.task
        self.fold = args.fold
        self.root_dir = f'/home/ylindq/Data/ISIC-2018/'
        self.result_dir = f'/home/ylindq/Experiment/ISIC-2018/{self.dataset}/'
        self.model['name'] = args.name
        self.model['pretrained'] = args.pretrained
        self.model['fix_params'] = args.fix_params
        self.model['in_c'] = args.in_c

        # --- training params --- #
        self.train['save_dir'] = '{:s}/{:s}/{:s}/fold_{:d}'.format(self.result_dir, self.task, self.model['name'], self.fold)  # path to save results
        self.train['input_size'] = args.train_input_size
        self.train['train_epochs'] = args.train_train_epochs
        self.train['batch_size'] = args.train_batch_size
        self.train['checkpoint_freq'] = args.train_checkpoint_freq
        self.train['lr'] = args.train_lr
        self.train['weight_decay'] = args.train_weight_decay
        self.train['log_interval'] = args.train_log_interval
        self.train['workers'] = args.train_workers
        self.train['gpus'] = args.train_gpus

        # --- resume training --- #
        self.train['start_epoch'] = args.train_start_epoch
        self.train['checkpoint'] = args.train_checkpoint
        self.train['seed'] = args.train_seed

        # --- test parameters --- #
        self.test['test_epoch'] = args.test_test_epoch
        self.test['gpus'] = args.test_gpus
        self.test['save_flag'] = args.test_save_flag
        self.test['patch_size'] = args.test_patch_size
        self.test['overlap'] = args.test_overlap
        self.test['batch_size'] = args.test_batch_size
        self.test['save_dir'] = '{:s}/test_results'.format(self.train['save_dir'])
        self.test['checkpoint_dir'] = '{:s}/checkpoints/'.format(self.train['save_dir'])
        self.test['model_path'] = '{:s}/checkpoint_{:d}.pth.tar'.format(self.test['checkpoint_dir'], self.test['test_epoch'])

        # --- post processing --- #
        self.post['min_area'] = 20  # minimum area for an object

        # define data transforms for training
        self.transform['train'] = A.Compose([
            A.Resize(self.train['input_size'][1], self.train['input_size'][0]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        self.transform['val'] = A.Compose([
            # A.Resize(height=self.train['input_size'][0], width=self.train['input_size'][1]),
            A.Resize(self.train['input_size'][1], self.train['input_size'][0]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        self.transform['test'] = A.Compose([
            # A.Resize(height=self.train['input_size'][0], width=self.train['input_size'][1]),
            A.Resize(self.train['input_size'][1], self.train['input_size'][0]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        if not os.path.exists(self.train['save_dir']):
            os.makedirs(self.train['save_dir'], exist_ok=True)
        if not os.path.exists(self.test['checkpoint_dir']):
            os.makedirs(self.test['checkpoint_dir'], exist_ok=True)
        if not os.path.exists(self.test['save_dir']):
            os.makedirs(self.test['save_dir'], exist_ok=True)


    def save_options(self):
        if self.isTrain:
            filename = '{:s}/train_options.txt'.format(self.train['save_dir'])
        else:
            filename = '{:s}/test_options.txt'.format(self.test['save_dir'])
        file = open(filename, 'w')
        # groups = ['model', 'train', 'transform'] if self.isTrain else ['model', 'test', 'post', 'transform']
        groups = ['model', 'train', ] if self.isTrain else ['model', 'test', 'post', ]

        file.write("# ---------- Options ---------- #")
        file.write('\ndataset: {:s}\n'.format(self.dataset))
        file.write('isTrain: {}\n'.format(self.isTrain))
        for group, options in self.__dict__.items():
            if group not in groups:
                continue
            file.write('\n\n-------- {:s} --------\n'.format(group))
            if group == 'transform':
                for name, val in options.items():
                    if (self.isTrain and name != 'test') or (not self.isTrain and name == 'test'):
                        file.write("{:s}:\n".format(name))
                        for t_name, t_val in val.items():
                            file.write("\t{:s}: {:s}\n".format(t_name, repr(t_val)))
            else:
                for name, val in options.items():
                    file.write("{:s} = {:s}\n".format(name, repr(val)))
        file.close()




