import torch
import random
from PIL import Image, ImageOps
import numpy as np
import numbers

def get_transforms(p):
    """ data transforms for train, validation and test
        p: transform dictionary
    """
    t_list = list()
    if 'random_resize' in p:
        t_list.append(RandomResize(p['random_resize'][0], p['random_resize'][1]))

    if 'horizontal_flip' in p:
        t_list.append(RandomHorizontalFlip())

    if 'vertical_flip' in p:
        t_list.append(RandomVerticalFlip())

    if 'random_affine' in p:
        t_list.append(RandomAffine(p['random_affine']))

    if 'random_rotation' in p:
        t_list.append(RandomRotation(p['random_rotation']))

    if 'random_crop' in p:
        t_list.append(RandomCrop(p['random_crop']))

    if 'label_encoding' in p:
        t_list.append(LabelEncoding(p['label_encoding']))

    if 'to_tensor' in p:
        t_list.append(ToTensor(p['to_tensor']))

    if 'normalize' in p:
        t_list.append(Normalize(mean=p['normalize'][0], std=p['normalize'][1]))

    return Compose(t_list)


class Compose(object):
    """ Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, imgs):
        for t in self.transforms:
            imgs = t(imgs)
        return imgs


class ToTensor(object):
    """ Convert (img, label) of type ``PIL.Image`` or ``numpy.ndarray`` to tensors.
    Converts img of type PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    Converts label of type PIL.Image or numpy.ndarray (H x W) in the range [0, 255]
    to a torch.LongTensor of shape (H x W) in the range [0, 255].
    """
    def __init__(self, index=1):
        self.index = index  # index to distinguish between images and labels

    def __call__(self, imgs):
        """
        Args:
            imgs (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if len(imgs) < self.index:
            raise ValueError('The number of images is smaller than separation index!')

        pics = []

        # process image
        for i in range(0, self.index):
            img = imgs[i]
            if isinstance(img, np.ndarray):
                # handle numpy array
                pic = torch.from_numpy(img.transpose((2, 0, 1)))
                # backward compatibility
                pics.append(pic.float().div(255))

            # handle PIL Image
            if img.mode == 'I':
                pic = torch.from_numpy(np.array(img, np.int32, copy=False))
            elif img.mode == 'I;16':
                pic = torch.from_numpy(np.array(img, np.int16, copy=False))
            else:
                pic = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
            # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
            if img.mode == 'YCbCr':
                nchannel = 3
            elif img.mode == 'I;16':
                nchannel = 1
            else:
                nchannel = len(img.mode)
            pic = pic.view(img.size[1], img.size[0], nchannel)
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            pic = pic.transpose(0, 1).transpose(0, 2).contiguous()
            if isinstance(pic, torch.ByteTensor):
                pics.append(pic.float().div(255))
            else:
                pics.append(pic)

        # process labels:
        for i in range(self.index, len(imgs)):
            # process label
            label = imgs[i]
            if isinstance(label, np.ndarray):
                # handle numpy array
                label_tensor = torch.from_numpy(label)
                # backward compatibility
                pics.append(label_tensor.long())

            # handle PIL Image
            if label.mode == 'I':
                label_tensor = torch.from_numpy(np.array(label, np.int32, copy=False))
            elif label.mode == 'I;16':
                label_tensor = torch.from_numpy(np.array(label, np.int16, copy=False))
            else:
                label_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(label.tobytes()))
            # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
            if label.mode == 'YCbCr':
                nchannel = 3
            elif label.mode == 'I;16':
                nchannel = 1
            else:
                nchannel = len(label.mode)
            label_tensor = label_tensor.view(label.size[1], label.size[0], nchannel)
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            label_tensor = label_tensor.transpose(0, 1).transpose(0, 2).contiguous()
            # label_tensor = label_tensor.view(label.size[1], label.size[0])
            pics.append(label_tensor.long())

        return tuple(pics)


class Normalize(object):
    """ Normalize an tensor image with mean and standard deviation.
    Given mean and std, will normalize each channel of the torch.*Tensor,
     i.e. channel = (channel - mean) / std
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    ** only normalize the first image, keep the target image unchanged
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensors):
        """
        Args:
            tensors (Tensor): Tensor images of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        tensors = list(tensors)
        for t, m, s in zip(tensors[0], self.mean, self.std):
            t.sub_(m).div_(s)
        return tuple(tensors)


class RandomCrop(object):
    """Crop the given PIL.Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __init__(self, size, padding=0, fill_val=(0,)):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.fill_val = fill_val

    def __call__(self, imgs):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        pics = []

        w, h = imgs[0].size
        th, tw = self.size
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for k in range(len(imgs)):
            img = imgs[k]
            if self.padding > 0:
                img = ImageOps.expand(img, border=self.padding, fill=self.fill_val[k])

            if w == tw and h == th:
                pics.append(img)
                continue

            pics.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return tuple(pics)


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, imgs):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        pics = []
        if random.random() < 0.5:
            for img in imgs:
                pics.append(img.transpose(Image.FLIP_LEFT_RIGHT))
            return tuple(pics)
        else:
            return imgs


class RandomVerticalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, imgs):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        pics = []
        if random.random() < 0.5:
            for img in imgs:
                pics.append(img.transpose(Image.FLIP_TOP_BOTTOM))
            return tuple(pics)
        else:
            return imgs


class RandomRotation(object):
    """Rotate the image by angle.
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(self, degrees, resample=Image.BILINEAR, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, imgs):
        """
            imgs (PIL Image): Images to be rotated.
        Returns:
            PIL Image: Rotated image.
        """

        angle = self.get_params(self.degrees)

        pics = []
        for img in imgs:
            pics.append(img.rotate(angle, self.resample, self.expand, self.center))

        # process the binary label
        # pics[1] = pics[1].point(lambda p: p > 127.5 and 255)

        return tuple(pics)


class RandomResize(object):
    """Randomly Resize the input PIL Image using a scale of lb~ub.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, lb=0.5, ub=1.5, interpolation=Image.BILINEAR):
        self.lb = lb
        self.ub = ub
        self.interpolation = interpolation

    def __call__(self, imgs):
        """
        Args:
            imgs (PIL Images): Images to be scaled.
        Returns:
            PIL Images: Rescaled images.
        """

        for img in imgs:
            if not isinstance(img, Image.Image):
                raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

        scale = random.uniform(self.lb, self.ub)
        # print scale

        w, h = imgs[0].size
        ow = int(w * scale)
        oh = int(h * scale)

        if scale < 1:
            padding_l = (w - ow)//2
            padding_t = (h - oh)//2
            padding_r = w - ow - padding_l
            padding_b = h - oh - padding_t
            padding = (padding_l, padding_t, padding_r, padding_b)

        pics = []
        for i in range(len(imgs)):
            img = imgs[i]
            img = img.resize((ow, oh), self.interpolation)
            if scale < 1:
                img = ImageOps.expand(img, border=padding, fill=0)
            pics.append(img)

        return tuple(pics)


class RandomAffine(object):
    """ Transform the input PIL Image using a random affine transformation
        The parameters of an affine transformation [a, b, c=0
                                                    d, e, f=0]
        are generated randomly according to the bound, and there is no translation
        (c=f=0)
    Args:
        bound: the largest possible deviation of random parameters
    """

    def __init__(self, bound):
        if bound < 0 or bound > 0.5:
            raise ValueError("Bound is invalid, should be in range [0, 0.5)")

        self.bound = bound

    def __call__(self, imgs):
        img = imgs[0]
        x, y = img.size

        a = 1 + 2 * self.bound * (random.random() - 0.5)
        b = 2 * self.bound * (random.random() - 0.5)
        d = 2 * self.bound * (random.random() - 0.5)
        e = 1 + 2 * self.bound * (random.random() - 0.5)

        # correct the transformation center to image center
        c = -a * x / 2 - b * y / 2 + x / 2
        f = -d * x / 2 - e * y / 2 + y / 2

        trans_matrix = [a, b, c, d, e, f]

        pics = []
        for img in imgs:
            pics.append(img.transform((x, y), Image.AFFINE, trans_matrix))

        return tuple(pics)


class LabelEncoding(object):
    """
    encode the 3-channel labels into one channel integer label map
    """
    def __init__(self, num_labels=1):
        self.num_labels = num_labels

    def __call__(self, imgs):
        assert self.num_labels < len(imgs)

        out_imgs = list(imgs[:-self.num_labels])
        image = imgs[0]  # input image
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        for i in range(-self.num_labels, 0):  # labels
            label = imgs[i]
            if not isinstance(label, np.ndarray):
                label = np.array(label)
            # print(len(label.shape))

            # ----- for estimated pixel-level label
            # new_label = np.ones((label.shape[0], label.shape[1]), dtype=np.uint8) * 2
            # new_label[label[:, :, 0] > 255 * 0.3] = 0
            # new_label[label[:, :, 1] > 255 * 0.5] = 1
            # new_label[(image[:, :, 0] == 0) * (image[:, :, 1] == 0) * (image[:, :, 2] == 0)] = 0
            # new_label = Image.fromarray(new_label.astype(np.uint8))
            # out_imgs.append(new_label)
            # ----- for binary label
            new_label = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
            new_label[label[:, :] > 255 * 0.3] = 1
            new_label = Image.fromarray(new_label.astype(np.uint8))
            out_imgs.append(new_label)

        return tuple(out_imgs)
