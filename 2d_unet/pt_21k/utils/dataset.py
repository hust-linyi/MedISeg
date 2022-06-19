from email.contentmanager import raw_data_manager
from genericpath import exists
import torch
import torch.utils.data as data
import os
from PIL import Image, ImageEnhance
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter, median_filter
import numpy as np
import pandas as pd
import random
from glob import glob
import math
import cv2


def data_split(filenames, fold=0, phase='train'):
    # for 5 fold cross validation
    validnum = int(len(filenames) * 0.2)
    valstart = fold * validnum
    valend = (fold + 1) * validnum
    if phase == 'train':
        filenames = np.concatenate([filenames[:valstart], filenames[valend:]], axis=0).tolist()
    else:
        filenames = filenames[valstart:valend]
    return filenames


# class DataFolder(data.Dataset):
#     def __init__(self, root_dir, phase, fold, data_transform=None):
#         """
#         :param root_dir: 
#         :param data_transform: data transformations
#         """
#         super(DataFolder, self).__init__()
#         self.data_transform = data_transform
#         self.phase = phase
#         self.mask_suffix = '_segmentation'
#         self.img_dir = os.path.join(root_dir, 'IMG')
#         self.mask_dir = os.path.join(root_dir, 'MASK')
#         self.ids = [os.path.splitext(file)[0] for file in os.listdir(self.img_dir) if file.endswith('.jpg')]
#         self.ids = data_split(self.ids, fold=fold, phase=phase)

#     def __len__(self):
#         return len(self.ids)

#     def __getitem__(self, idx):
#         name = self.ids[idx]
#         img = cv2.imread(os.path.join(self.img_dir, name + '.jpg'))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         mask = cv2.imread(os.path.join(self.mask_dir, name + self.mask_suffix + '.png'), cv2.IMREAD_GRAYSCALE)
#         mask = mask / 255.0

#         assert img is not None and mask is not None, 'Image or mask is None, idx: {}'.format(idx)
#         assert img.size == 3 * mask.size, \
#             "Image and mask size don't match, but got {} and {}, idx: {}".format(img.size, mask.size, idx)
        
#         if self.data_transform is not None:
#             transformed = self.data_transform(image=img, mask=mask)
#             img = transformed['image']
#             mask = transformed['mask']
#         return img, mask, name


# class DataFolder(data.Dataset):
#     def __init__(self, root_dir, phase, fold, data_transform=None):
#         """
#         :param root_dir: 
#         :param data_transform: data transformations
#         """
#         super(DataFolder, self).__init__()
#         self.data_transform = data_transform
#         self.phase = phase
#         self.mask_suffix = '_segmentation'
#         self.img_dir = os.path.join(root_dir, 'IMG')
#         self.mask_dir = os.path.join(root_dir, 'MASK')
#         self.ids = [os.path.splitext(file)[0] for file in os.listdir(self.img_dir) if file.endswith('.jpg')]
#         self.ids = data_split(self.ids, fold=fold, phase=phase)
#         self.imgs, self.masks, self.filenames = self.load_dataset()

#     def __len__(self):
#         return len(self.ids)

#     def __getitem__(self, idx):
#         img, mask, name = self.imgs[idx], self.masks[idx], self.filenames[idx]
        
#         if self.data_transform is not None:
#             transformed = self.data_transform(image=img, mask=mask)
#             img = transformed['image']
#             mask = transformed['mask']
#         return img, mask, name
    
#     def load_dataset(self):
#         imgs = []
#         masks = []
#         filenames = []
#         for name in self.ids:
#             img = cv2.imread(os.path.join(self.img_dir, name + '.jpg'))
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             mask = cv2.imread(os.path.join(self.mask_dir, name + self.mask_suffix + '.png'), cv2.IMREAD_GRAYSCALE)
#             mask = mask / 255.0
#             imgs.append(img)
#             masks.append(mask)
#             filenames.append(name)
#         return imgs, masks, filenames

#     def data_split(filenames, fold=0, phase='train'):
#         # for 5 fold cross validation
#         validnum = int(len(filenames) * 0.2)
#         valstart = fold * validnum
#         valend = (fold + 1) * validnum
#         if phase == 'train':
#             filenames = np.concatenate([filenames[:valstart], filenames[valend:]], axis=0).tolist()
#         else:
#             filenames = filenames[valstart:valend]
#         return filenames


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
        self.imgs, self.masks, self.filenames = self.load_dataset()

    def __len__(self):
        return self.filenames.shape[0]

    def __getitem__(self, idx):
        img, mask, name = self.imgs[idx], self.masks[idx], self.filenames[idx]
        
        if self.data_transform is not None:
            transformed = self.data_transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']
        return img, mask, name
    
    def load_dataset(self):
        imgs = np.load(os.path.join(self.root_dir, 'NumpyData', 'img.npy'))
        mask = np.load(os.path.join(self.root_dir, 'NumpyData', 'mask.npy'))
        filenames = np.load(os.path.join(self.root_dir, 'NumpyData', 'filename.npy'))

        validnum = int(len(filenames) * 0.2)
        valstart = self.fold * validnum
        valend = (self.fold + 1) * validnum

        if self.phase == 'train':
            filenames = np.concatenate([filenames[:valstart], filenames[valend:]], axis=0)
            imgs = np.concatenate([imgs[:valstart], imgs[valend:]], axis=0)
            masks = np.concatenate([mask[:valstart], mask[valend:]], axis=0)
        else:
            filenames = filenames[valstart:valend]
            imgs = imgs[valstart:valend]
            masks = mask[valstart:valend]
        return imgs, masks, filenames