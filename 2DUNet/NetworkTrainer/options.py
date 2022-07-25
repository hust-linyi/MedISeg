import os
import argparse
from NetworkTrainer.dataloaders.get_transform import get_transform


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
        parser.add_argument('--dataset', type=str, default='isic2018', help='isic2018 or conic')
        parser.add_argument('--task', type=str, default='baseline', help='')
        parser.add_argument('--fold', type=int, default=0, help='0-4, five fold cross validation')
        parser.add_argument('--name', type=str, default='res50', help='res34, res50, res101, res152')
        parser.add_argument('--pretrained', type=bool, default=False, help='True or False')
        parser.add_argument('--in-c', type=int, default=3, help='input channel')
        parser.add_argument('--input-size', type=list, default=[256,256], help='input size of the image')
        parser.add_argument('--train-train-epochs', type=int, default=200, help='number of training epochs')
        parser.add_argument('--train-batch-size', type=int, default=32, help='batch size')
        parser.add_argument('--train-checkpoint-freq', type=int, default=500, help='epoch to save checkpoints')
        parser.add_argument('--train-lr', type=float, default=3e-4, help='initial learning rate')
        parser.add_argument('--train-weight-decay', type=float, default=1e-5, help='weight decay')
        parser.add_argument('--train-workers', type=int, default=16, help='number of workers to load images')
        parser.add_argument('--train-gpus', type=list, default=[0, ], help='select gpu devices')
        parser.add_argument('--train-start-epoch', type=int, default=0, help='start epoch')
        parser.add_argument('--train-checkpoint', type=str, default='', help='checkpoint')
        parser.add_argument('--train-seed', type=int, default=2022, help='bn or in')
        parser.add_argument('--train-loss', type=str, default='ce', help='save directory')
        parser.add_argument('--train-deeps', type=bool, default=False, help='save directory')
        parser.add_argument('--test-test-epoch', type=int, default=0, help='test epoch')
        parser.add_argument('--test-gpus', type=list, default=[0, ], help='select gpu devices')
        parser.add_argument('--test-save-flag', type=bool, default=False, help='True or False')
        parser.add_argument('--test-batch-size', type=int, default=4, help='batch size')
        parser.add_argument('--test-flip', type=bool, default=False, help='True or False')       
        parser.add_argument('--test-rotate', type=bool, default=False, help='True or False')       
        parser.add_argument('--post-abl', type=bool, default=False, help='True or False, post processing')
        parser.add_argument('--post-rsa', type=bool, default=False, help='True or False, post processing')


        args = parser.parse_args()

        self.dataset = args.dataset
        self.task = args.task
        self.fold = args.fold
        # check if the root directory exists
        home_dir = '/home/ylindq'
        if not os.path.exists(home_dir):
            home_dir = '/newdata/ianlin/'
        self.root_dir = home_dir + '/Data/ISIC-2018'
        self.result_dir = home_dir + f'/Experiment/ISIC-2018/{self.dataset}/'
        if 'conic' in self.dataset:
            self.root_dir = home_dir + '/Data/CoNIC_Challenge'
            self.result_dir = home_dir + f'/Experiment/CoNIC_Challenge/{self.dataset}/'
        self.model['name'] = args.name
        self.model['pretrained'] = args.pretrained
        self.model['in_c'] = args.in_c
        self.model['input_size'] = args.input_size

        # --- training params --- #
        self.train['save_dir'] = '{:s}/{:s}/{:s}/fold_{:d}'.format(self.result_dir, self.task, self.model['name'], self.fold)  # path to save results
        self.train['train_epochs'] = args.train_train_epochs
        self.train['batch_size'] = args.train_batch_size
        self.train['checkpoint_freq'] = args.train_checkpoint_freq
        self.train['lr'] = args.train_lr
        self.train['weight_decay'] = args.train_weight_decay
        self.train['workers'] = args.train_workers
        self.train['gpus'] = args.train_gpus
        self.train['seed'] = args.train_seed
        self.train['loss'] = args.train_loss
        self.train['deeps'] = args.train_deeps

        # --- resume training --- #
        self.train['start_epoch'] = args.train_start_epoch
        self.train['checkpoint'] = args.train_checkpoint

        # --- test parameters --- #
        self.test['test_epoch'] = args.test_test_epoch
        self.test['gpus'] = args.test_gpus
        self.test['save_flag'] = args.test_save_flag
        self.test['batch_size'] = args.test_batch_size
        self.test['flip'] = args.test_flip
        self.test['rotate'] = args.test_rotate
        self.test['save_dir'] = '{:s}/test_results'.format(self.train['save_dir'])
        self.test['checkpoint_dir'] = '{:s}/checkpoints/'.format(self.train['save_dir'])
        self.test['model_path'] = '{:s}/checkpoint_{:d}.pth.tar'.format(self.test['checkpoint_dir'], self.test['test_epoch'])

        # --- post processing --- #
        self.post['abl'] = args.post_abl
        self.post['rsa'] = args.post_rsa

        # define data transforms for training
        self.transform['train'] = get_transform(self, 'train')
        self.transform['val'] = get_transform(self, 'val')
        self.transform['test'] = get_transform(self, 'test')



    def save_options(self):
        if not os.path.exists(self.train['save_dir']):
            os.makedirs(self.train['save_dir'], exist_ok=True)
        if not os.path.exists(self.test['checkpoint_dir']):
            os.makedirs(self.test['checkpoint_dir'], exist_ok=True)
        if not os.path.exists(self.test['save_dir']):
            os.makedirs(self.test['save_dir'], exist_ok=True)

        if self.isTrain:
            filename = '{:s}/train_options.txt'.format(self.train['save_dir'])
        else:
            filename = '{:s}/test_options.txt'.format(self.test['save_dir'])
        file = open(filename, 'w')
        groups = ['model', 'test', 'post', 'transform']
        # groups = ['model', 'train', ] if self.isTrain else ['model', 'test', 'post', ]

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
                        for t_val in val:
                            file.write("\t{:s}\n".format(t_val.__class__.__name__))
            else:
                for name, val in options.items():
                    file.write("{:s} = {:s}\n".format(name, repr(val)))
        file.close()



