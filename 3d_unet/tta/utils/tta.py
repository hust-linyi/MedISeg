import torch
import numpy as np

class TTA():
    def __init__(self, if_tta):
        self.if_tta = if_tta

    def img_list(self, img):
        out = []
        out.append(img)
        if not self.if_tta:
            return out
        # apply flip
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



    

