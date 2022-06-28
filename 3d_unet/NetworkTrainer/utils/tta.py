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



    

