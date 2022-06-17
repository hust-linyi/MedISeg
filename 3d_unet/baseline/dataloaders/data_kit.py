import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import itertools
from torch.utils.data.sampler import Sampler


class DataFolder(Dataset):
    """ Kit19 Dataset """
    def __init__(self, root_dir, phase, fold, data_transform=None):
        self.root_dir = root_dir
        self.data_transform = data_transform
        self.image_list = [item.replace('','') for item in self.root_dir if item.endswith('_image.npy')]

        valnum = int(len(self.image_list) * 0.2)
        valstart = fold * valnum
        valend = (fold + 1) * valnum
        self.image_list = np.concatenate([self.image_list[:valstart], self.image_list[valend:]], axis=0)
        if phase == 'train':
            valnum = int(len(self.image_list) * 0.1)
            self.image_list = self.image_list[valnum:]
        elif phase == 'val':
            valnum = int(len(self.image_list) * 0.1)
            self.image_list = self.image_list[:valnum]
        print(f"total {len(self.image_list)} samples for {phase}")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image = np.load(os.path.join(self.root_dir, image_name + "_image.npy"))
        label = np.load(os.path.join(self.root_dir, image_name + "_label.npy"))
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


class SelectedCrop(object):
    def __init__(self, output_size, oversample_foreground_percent=0.3):
        self.output_size = output_size
        self.percent = oversample_foreground_percent

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if np.random.random() < self.percent:
            pixels = np.argwhere(label != 0)
            if len(pixels) == 0:
                return RandomCrop(self.output_size)(sample)
            else:
                selected_pixel = pixels[np.random.choice(len(pixels))]
                pw = self.output_size[0] // 2 + 1
                ph = self.output_size[1] // 2 + 1
                pd = self.output_size[2] // 2 + 1

                image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
                label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
                bbox_x_lb = selected_pixel[0]
                bbox_y_lb = selected_pixel[1]
                bbox_z_lb = selected_pixel[2]

                label = label[bbox_x_lb:bbox_x_lb + self.output_size[0], bbox_y_lb:bbox_y_lb + self.output_size[1], bbox_z_lb:bbox_z_lb + self.output_size[2]]
                image = image[bbox_x_lb:bbox_x_lb + self.output_size[0], bbox_y_lb:bbox_y_lb + self.output_size[1], bbox_z_lb:bbox_z_lb + self.output_size[2]]
                return {'image': image, 'label': label}

        else:
            return RandomCrop(self.output_size)(sample)


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return {'image': image, 'label': label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'image': image, 'label': label}
class RandomMirroring(object):
    def __init__(self, axes=(0, 1, 2)):
        self.axes = axes
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if 0 in self.axes  and np.random.uniform() < 0.5:
            image[:] = image[::-1]
            label[:] = label[::-1]
        if 1 in self.axes  and np.random.uniform() < 0.5:
            image[:, :] = image[:,::-1]
            label[:, :] = label[:,::-1]
        if 2 in self.axes  and np.random.uniform() < 0.5:
            image[:, :, :] = image[:, :, ::-1]
            label[:, :, :] = label[:, :, ::-1]
        return {'image': image, 'label': label}

class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}

class GammaAdjust(object):
    def __init__(self, gamma_range=(0.5, 2), epsilon=1e-7,retain_stats = False):
        self.gamma_range = gamma_range
        self.epsilon = epsilon
        self.retain_stats = retain_stats

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.retain_stats:
            mn = image.mean()
            sd = image.std()
        if np.random.random() < 0.5 and self.gamma_range[0] < 1:
            gamma = np.random.uniform(self.gamma_range[0], 1)
        else:
            gamma = np.random.uniform(max(self.gamma_range[0], 1), self.gamma_range[1])
        minm = image.min()
        rnge = image.max() - minm
        image = np.power(((image - minm) / float(rnge + self.epsilon)), gamma) * rnge + minm
        if self.retain_stats:
            image = image - image.mean() + mn
            image = image / (image.std() + 1e-8) * sd
        return {'image': image, 'label': label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def to_onehot(self, label, num_class):
        out = np.zeros_like(label)
        out = np.repeat(out, num_class, axis=0)
        for i in range(num_class):
            out[i, ...] = label[0, ...] == i
        return out

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        label = label.reshape(1, label.shape[0], label.shape[1], label.shape[2]).astype(np.float32)
        # label = self.to_onehot(label, 11)
        return {'image': torch.from_numpy(image), 'label': torch.from_numpy(label)}

# class TwoStreamBatchSampler(Sampler):
#     """Iterate two sets of indices

#     An 'epoch' is one iteration through the primary indices.
#     During the epoch, the secondary indices are iterated through
#     as many times as needed.
#     """
#     def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
#         self.primary_indices = primary_indices
#         self.secondary_indices = secondary_indices
#         self.secondary_batch_size = secondary_batch_size
#         self.primary_batch_size = batch_size - secondary_batch_size

#         assert len(self.primary_indices) >= self.primary_batch_size > 0
#         assert len(self.secondary_indices) >= self.secondary_batch_size > 0

#     def __iter__(self):
#         primary_iter = iterate_once(self.primary_indices)
#         secondary_iter = iterate_eternally(self.secondary_indices)
#         return (
#             primary_batch + secondary_batch
#             for (primary_batch, secondary_batch)
#             in zip(grouper(primary_iter, self.primary_batch_size),
#                     grouper(secondary_iter, self.secondary_batch_size))
#         )

#     def __len__(self):
#         return len(self.primary_indices) // self.primary_batch_size

# def iterate_once(iterable):
#     return np.random.permutation(iterable)


# def iterate_eternally(indices):
#     def infinite_shuffles():
#         while True:
#             yield np.random.permutation(indices)
#     return itertools.chain.from_iterable(infinite_shuffles())


# def grouper(iterable, n):
#     "Collect data into fixed-length chunks or blocks"
#     # grouper('ABCDEFG', 3) --> ABC DEF"
#     args = [iter(iterable)] * n
#     return zip(*args)
