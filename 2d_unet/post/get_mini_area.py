import numpy as np
import os

def get_mini_area_conic():
    label_path = '/newdata/ianlin/Data/CoNIC_Challenge/labels.npy'
    label = np.load(label_path)
    label = label[...,0]
    # debug
    label = label[:100]
    import ipdb; ipdb.set_trace()
    mini_area = 1e5
    for i in range(label.shape[0]):
        _label = label[i]
        for j in range(1, np.max(_label)):
            area = np.sum(_label==j)
            print(f'{j}:{area}')
            if area < mini_area:
                mini_area = area
    print(mini_area)

get_mini_area_conic()