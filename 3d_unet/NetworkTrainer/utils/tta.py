import torch
import numpy as np

class TTA():
    def __init__(self, if_flip=False, if_rot=False):
        self.if_flip = if_flip
        self.if_rot = if_rot

    def img_list(self, img):
        # for Kit, the shape is (x, y, z)
        out = []
        out.append(img)
        if self.if_flip:
        # apply flip
            for i in range(3):
                out.append(np.flip(img, axis=i))
        if self.if_rot:
            # apply rotation
            for i in range(3):
                out.append(np.rot90(img, k=(i+1), axes=(0,1)))
        return out
    
    def img_list_inverse(self, img_list):
        # for Kit, the shape is (c=3, x, y, z)
        out = [img_list[0]]
        if self.if_flip:
            # apply flip
            for i in range(3):
                out.append(np.flip(img_list[i+1], axis=(i+1)))
        if self.if_rot:
            # apply rotation
            for i in range(3):
                out.append(np.rot90(img_list[i+4], k=-(i+1), axes=(1,2)))
        return out


if __name__=='__main__':
    # test TTA
    import os
    import nibabel as nib
    data_dir = '/newdata/ianlin/Data/KIT-19/yeung/preprocess'
    case_id = 67
    img = np.load(os.path.join(data_dir, f'case_000{case_id}_image.npy'))
    mask = np.load(os.path.join(data_dir, f'case_000{case_id}_label.npy'))

    tta = TTA(if_flip=True, if_rot=True)
    img_list = tta.img_list(img)
    mask_list = tta.img_list(mask)
    img_list = [f[np.newaxis, ...] for f in img_list]
    mask_list = [f[np.newaxis, ...] for f in mask_list]
    img_list_inverse = tta.img_list_inverse(img_list)
    mask_list_inverse = tta.img_list_inverse(mask_list)
    for i in range(len(img_list_inverse)):
        _img = img_list_inverse[i]
        _mask = mask_list_inverse[i].squeeze()
        if (_img != img).any():
            print('bug, img')
        if (_mask != mask).any():
            print('bug, mask')
        # nib.save(nib.Nifti1Image(_img, np.eye(4)), f'img_{i}.nii.gz')
        # nib.save(nib.Nifti1Image(_mask, np.eye(4)), f'mask_{i}.nii.gz')
    


    

