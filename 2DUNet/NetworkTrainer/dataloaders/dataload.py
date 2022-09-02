from cv2 import phase
import torch.utils.data as data
import os
import numpy as np
import pandas as pd

class DataFolder(data.Dataset):
    def __init__(self, root_dir, phase, fold, gan_aug=False, data_transform=None):
        """
        :param root_dir: 
        :param data_transform: data transformations
        :param phase: train, val, test
        :param fold: fold number, 0, 1, 2, 3, 4
        :param gan_aug: whether to use gan augmentation
        """
        super(DataFolder, self).__init__()
        self.data_transform = data_transform
        self.gan_aug = gan_aug
        self.phase = phase
        self.fold = fold 
        self.root_dir = root_dir
        self.imgs, self.masks, self.filenames = self.load_dataset(os.path.join(self.root_dir, 'NumpyData'))
        self.imgs_aug, self.masks_aug, _ = self.load_dataset(os.path.join(self.root_dir, 'aug'))      

    def __len__(self):
        return self.filenames.shape[0]

    def __getitem__(self, idx):
        img, mask, name = self.imgs[idx], self.masks[idx], self.filenames[idx]
        if self.gan_aug and np.random.rand() < 0.2:
            img = self.imgs_aug[idx]
            mask = self.masks_aug[idx]
        
        if self.data_transform is not None:
            transformed = self.data_transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']
        return {'image': img, 'label': mask, 'name': name}
    
    def load_dataset(self, data_dir):
        imgs = np.load(os.path.join(data_dir, 'img.npy'))
        mask = np.load(os.path.join(data_dir, 'mask.npy'))
        filenames = np.load(os.path.join(data_dir, 'filename.npy'))
        testnum = int(len(filenames) * 0.2)
        teststart = self.fold * testnum
        testend = (self.fold + 1) * testnum

        if self.phase == 'test':
            filenames = filenames[teststart:testend]
            imgs = imgs[teststart:testend]
            masks = mask[teststart:testend]
        else:
            filenames = np.concatenate([filenames[:teststart], filenames[testend:]], axis=0)
            imgs = np.concatenate([imgs[:teststart], imgs[testend:]], axis=0)
            masks = np.concatenate([mask[:teststart], mask[testend:]], axis=0)
            valnum = int(len(filenames) * 0.2)
            if self.phase == 'train':
                imgs, masks, filenames = imgs[valnum:], masks[valnum:], filenames[valnum:]
            elif self.phase == 'val':
                imgs, masks, filenames = imgs[:valnum], masks[:valnum], filenames[:valnum]
            else:
                raise ValueError('phase should be train or val or test')
        return imgs, masks, filenames
