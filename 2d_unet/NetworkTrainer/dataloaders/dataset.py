from cv2 import phase
import torch.utils.data as data
import os
import numpy as np
import pandas as pd

class DataFolder(data.Dataset):
    def __init__(self, root_dir, phase, fold, data_transform=None):
        """
        :param root_dir: 
        :param data_transform: data transformations
        """
        super(DataFolder, self).__init__()
        self.data_transform = data_transform
        self.phase = phase
        self.fold = fold 
        self.root_dir = root_dir
        if 'ISIC' in root_dir:
            self.imgs, self.masks, self.filenames = self.load_dataset()
        elif 'CoNIC_Challenge' in root_dir:
            self.imgs, self.masks, self.filenames = self.load_dataset_conic()
        else:
            raise ValueError('root_dir should be ISIC or CoNIC_Challenge')

    def __len__(self):
        return self.filenames.shape[0]

    def __getitem__(self, idx):
        img, mask, name = self.imgs[idx], self.masks[idx], self.filenames[idx]
        
        if self.data_transform is not None:
            transformed = self.data_transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']
        return {'image': img, 'label': mask, 'name': name}
    
    def load_dataset(self):
        # Load ISIC2018 dataset from numpy files
        imgs = np.load(os.path.join(self.root_dir, 'NumpyData', 'img.npy'))
        mask = np.load(os.path.join(self.root_dir, 'NumpyData', 'mask.npy'))
        filenames = np.load(os.path.join(self.root_dir, 'NumpyData', 'filename.npy'))
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


    def load_dataset_conic(self):
        # Load CoNIC dataset from numpy files
        imgs = np.load(os.path.join(self.root_dir, 'images.npy'))
        mask = np.load(os.path.join(self.root_dir, 'labels.npy'))
        imgs = imgs.astype(np.uint8)
        mask = (mask[..., 1] > 0).astype(np.float32)
        filenames = pd.read_csv(os.path.join(self.root_dir, 'patch_info.csv')).values
        filenames = filenames[:, 0]
        # patients = [x.split('-')[0] for x in filenames]

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
