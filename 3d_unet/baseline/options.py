import os
import argparse
from torchvision import transforms
from dataloaders.data_kit import RandomCrop, CenterCrop, ToTensor
from dataloaders.get_transform import get_transform


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
        parser.add_argument('--dataset', type=str, default='kit19', help='dataset name')
        parser.add_argument('--task', type=str, default='baseline', help='')
        parser.add_argument('--fold', type=int, default=0, help='0-4, five fold cross validation')
        parser.add_argument('--pretrained', type=bool, default=False, help='True or False')
        parser.add_argument('--in-c', type=int, default=1, help='input channel')
        parser.add_argument('--patch-size', type=tuple, default=(96,96,96), help='input size of the image')
        parser.add_argument('--train-train-epochs', type=int, default=100, help='number of training epochs')
        parser.add_argument('--train-batch-size', type=int, default=2, help='batch size')
        parser.add_argument('--train-checkpoint-freq', type=int, default=20, help='epoch to save checkpoints')
        parser.add_argument('--train-lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--train-weight-decay', type=float, default=1e-4, help='weight decay')
        parser.add_argument('--train-workers', type=int, default=16, help='number of workers to load images')
        parser.add_argument('--train-gpus', type=list, default=[0, ], help='select gpu devices')
        parser.add_argument('--train-start-epoch', type=int, default=0, help='start epoch')
        parser.add_argument('--train-checkpoint', type=str, default='', help='checkpoint')
        parser.add_argument('--train-norm', type=str, default='bn', help='bn or in')
        parser.add_argument('--train-seed', type=str, default=2022, help='bn or in')
        parser.add_argument('--test-test-epoch', type=int, default=999, help='test epoch')
        parser.add_argument('--test-gpus', type=list, default=[0, ], help='select gpu devices')
        parser.add_argument('--test-save-flag', type=bool, default=True, help='True or False')
        args = parser.parse_args()

        self.dataset = args.dataset
        self.task = args.task
        self.fold = args.fold
        # check if the root directory exists
        home_dir = '/home/ylindq'
        if not os.path.exists(home_dir):
            home_dir = '/newdata/ianlin/'
        self.root_dir = home_dir + '/Data/KIT-19/yeung/preprocess'
        self.result_dir = home_dir + f'/Experiment/KIT19/{self.dataset}/'
        self.model['pretrained'] = args.pretrained
        self.model['in_c'] = args.in_c
        self.model['input_size'] = args.patch_size

        # --- training params --- #
        self.train['save_dir'] = '{:s}/{:s}/fold_{:d}'.format(self.result_dir, self.task, self.fold)  # path to save results
        self.train['train_epochs'] = args.train_train_epochs
        self.train['batch_size'] = args.train_batch_size
        self.train['checkpoint_freq'] = args.train_checkpoint_freq
        self.train['lr'] = args.train_lr
        self.train['weight_decay'] = args.train_weight_decay
        self.train['workers'] = args.train_workers
        self.train['gpus'] = args.train_gpus
        self.train['norm'] = args.train_norm
        self.train['seed'] = args.train_seed

        # --- resume training --- #
        self.train['start_epoch'] = args.train_start_epoch
        self.train['checkpoint'] = args.train_checkpoint

        # --- test parameters --- #
        self.test['test_epoch'] = args.test_test_epoch
        self.test['gpus'] = args.test_gpus
        self.test['save_flag'] = args.test_save_flag
        self.test['save_dir'] = '{:s}/test_results'.format(self.train['save_dir'])
        self.test['checkpoint_dir'] = '{:s}/checkpoints/'.format(self.train['save_dir'])
        self.test['model_path'] = '{:s}/checkpoint_{:d}.pth.tar'.format(self.test['checkpoint_dir'], self.test['test_epoch'])

        # --- post processing --- #
        self.post['min_area'] = 20  # minimum area for an object

        # define data transforms for training
        self.transform['train'] = get_transform(self, 'train')
        self.transform['val'] = get_transform(self, 'val')

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
        groups = ['model', 'train', ] if self.isTrain else ['model', 'test', 'post', 'transform']

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




