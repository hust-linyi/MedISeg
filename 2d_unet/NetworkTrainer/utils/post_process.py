import imp
import skimage.morphology as morph
import numpy as np
from scipy.ndimage import label

def abl(image: np.ndarray, for_which_classes: list, volume_per_voxel: float = None,
                                                   minimum_valid_object_size: dict = None):
    """
    removes all but the largest connected component, individually for each class
    :param image:
    :param for_which_classes: can be None. Should be list of int. Can also be something like [(1, 2), 2, 4].
    Here (1, 2) will be treated as a joint region, not individual classes (example LiTS here we can use (1, 2)
    to use all foreground classes together)
    :param minimum_valid_object_size: Only objects larger than minimum_valid_object_size will be removed. Keys in
    minimum_valid_object_size must match entries in for_which_classes
    :return:
    """
    if for_which_classes is None:
        for_which_classes = np.unique(image)
        for_which_classes = for_which_classes[for_which_classes > 0]
    assert 0 not in for_which_classes, "cannot remove background"

    if volume_per_voxel is None:
        volume_per_voxel = 1

    largest_removed = {}
    kept_size = {}
    for c in for_which_classes:
        if isinstance(c, (list, tuple)):
            c = tuple(c)  # otherwise it cant be used as key in the dict
            mask = np.zeros_like(image, dtype=bool)
            for cl in c:
                mask[image == cl] = True
        else:
            mask = image == c
        # get labelmap and number of objects
        lmap, num_objects = label(mask.astype(int))

        # collect object sizes
        object_sizes = {}
        for object_id in range(1, num_objects + 1):
            object_sizes[object_id] = (lmap == object_id).sum() * volume_per_voxel

        largest_removed[c] = None
        kept_size[c] = None

        if num_objects > 0:
            # we always keep the largest object. We could also consider removing the largest object if it is smaller
            # than minimum_valid_object_size in the future but we don't do that now.
            maximum_size = max(object_sizes.values())
            kept_size[c] = maximum_size

            for object_id in range(1, num_objects + 1):
                # we only remove objects that are not the largest
                if object_sizes[object_id] != maximum_size:
                    # we only remove objects that are smaller than minimum_valid_object_size
                    remove = True
                    if minimum_valid_object_size is not None:
                        remove = object_sizes[object_id] < minimum_valid_object_size[c]
                    if remove:
                        image[(lmap == object_id) & mask] = 0
                        if largest_removed[c] is None:
                            largest_removed[c] = object_sizes[object_id]
                        else:
                            largest_removed[c] = max(largest_removed[c], object_sizes[object_id])
    # return image, largest_removed, kept_size
    return image


def rsa(image: np.array, for_which_classes: list, volume_per_voxel: float = None, minimum_valid_object_size: dict = None):
    """
    Remove samll objects, smaller than minimum_valid_object_size, individually for each class
    :param image:
    :param for_which_classes: can be None. Should be list of int. Can also be something like [(1, 2), 2, 4].
    Here (1, 2) will be treated as a joint region, not individual classes (example LiTS here we can use (1, 2)
    to use all foreground classes together)
    :param minimum_valid_object_size: Only objects larger than minimum_valid_object_size will be removed. Keys in
    minimum_valid_object_size must match entries in for_which_classes
    :return:
    """
    if for_which_classes is None:
        for_which_classes = np.unique(image)
        for_which_classes = for_which_classes[for_which_classes > 0]
    assert 0 not in for_which_classes, "cannot remove background"

    if volume_per_voxel is None:
        volume_per_voxel = 1
    
    for c in for_which_classes:
        if isinstance(c, (list, tuple)):
            c = tuple(c)
            mask = np.zeros_like(image, dtype=bool)
            for cl in c:
                mask[image == cl] = True
        else:
            mask = image == c
        # get labelmap and number of objects
        lmap, num_objects = label(mask.astype(int))

        # collect object sizes
        object_sizes = {}
        for object_id in range(1, num_objects + 1):
            object_sizes[object_id] = (lmap == object_id).sum() * volume_per_voxel

        if num_objects > 0:
            # removing the largest object if it is smaller than minimum_valid_object_size.
            for object_id in range(1, num_objects + 1):
                # we only remove objects that are smaller than minimum_valid_object_size
                if object_sizes[object_id] < minimum_valid_object_size[c]:
                    image[(lmap == object_id) & mask] = 0
    
    return image


class TTA():
    def __init__(self, if_tta):
        # for ISIC, the shape is (b, c, h, w)
        # for Kit, the shape is (x, y, z)
        self.if_tta = if_tta

    def img_list(self, img):
        out = []
        out.append(img)
        if not self.if_tta:
            return out
        # apply flip
        import ipdb; ipdb.set_trace()
        for i in range(3):
            out.append(np.flip(img, axis=i))
        # apply rotation
        for i in range(1, 4):
            out.append(np.rot90(img, k=i))
        return out
    
    def img_list_inverse(self, img_list):
        out = [img_list[0]]
        if not self.if_tta:
            return img_list
        # apply flip
        for i in range(3):
            out.append(np.flip(img_list[i+1], axis=i))
        if len(img_list) > 4:
            # apply rotation
            for i in range(3):
                out.append(np.rot90(img_list[i+4], k=-(i+1), axes=(1,2)))
        return out



class TTA_2d():
    def __init__(self, flip=False, rotate=False):
        self.flip = flip
        self.rotate = rotate

    def img_list(self, img):
        # for ISIC, the shape is torch.size(b, c, h, w)
        img = img.detach().cpu().numpy()
        out = []
        out.append(img)
        if self.flip:
            # apply flip
            for i in range(2,4):
                out.append(np.flip(img, axis=i))
        if self.rotate:
            # apply rotation
            for i in range(1, 4):
                out.append(np.rot90(img, k=i, axes=(2,3)))
        return out
    
    def img_list_inverse(self, img_list):
        # for ISIC, the shape is numpy(b, h, w)
        out = [img_list[0]]
        if self.flip:
            # apply flip
            for i in range(2):
                out.append(np.flip(img_list[i+1], axis=i+1))
        if self.rotate:
            # apply rotation
            for i in range(3):
                out.append(np.rot90(img_list[i+3], k=-(i+1), axes=(1,2)))
        return out


if __name__ == '__main__':
    import torch
    a = torch.randn(4, 3, 256, 256)
    b = TTA_2d(True).img_list(a)
    print(b[0].shape)
    c = TTA_2d(True).img_list_inverse(b)
    print(c)