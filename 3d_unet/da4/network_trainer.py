import sys
sys.path.append('../')
from NetworkTrainer.network_trainer import NetworkTrainer
from data_kit_aug import DataFolder
from torch.utils.data import DataLoader
from torchvision import transforms

class NetworkTrainer_da3(NetworkTrainer):
    def set_dataloader(self):
        self.train_set = DataFolder(root_dir=self.opt.root_dir, phase='train', fold=self.opt.fold, data_transform=transforms.Compose(self.opt.transform['train']))
        self.val_set = DataFolder(root_dir=self.opt.root_dir, phase='val', data_transform=transforms.Compose(self.opt.transform['val']), fold=self.opt.fold)
        self.train_loader = DataLoader(self.train_set, batch_size=self.opt.train['batch_size'], shuffle=True, num_workers=self.opt.train['workers'])
        self.val_loader = DataLoader(self.val_set, batch_size=self.opt.train['batch_size'], shuffle=False, drop_last=False, num_workers=self.opt.train['workers'])
