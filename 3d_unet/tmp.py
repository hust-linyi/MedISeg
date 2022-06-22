"""
check the npy data size
"""
import os
import numpy as np
from rich import print
import time

def check_size(path):
    size = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            size += os.path.getsize(os.path.join(root, file))
    return size

def compress_npy(path):
    size_before = check_size(path)
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.gz') or file.endswith('gt.npy') or file.endswith('img.npy') or file.endswith('prob.npy'):
                os.remove(os.path.join(root, file))
            elif file.endswith('.npy'):
                # print(f'{os.path.join(root, file)}')
                data = np.load(os.path.join(root, file))
                np.savez_compressed(os.path.join(root, file.replace('.npy', '.npz')), data)
                os.remove(os.path.join(root, file))

    size_after = check_size(path)
    print(f'{path}, BEFORE: {size_before/1024/1024/1024:.2f} GB, AFTER: {size_after/1024/1024/1024:.2f} GB')


if __name__ == '__main__':
    data_dir = '/newdata/ianlin/Experiment/KIT19/kit19/'
    fold_list = os.listdir(data_dir)
    fold_list.remove('ensembleinit')
    for fold in fold_list:
        time1 = time.time()
        com_dir = os.path.join(data_dir, fold)
        compress_npy(com_dir)
        time2 = time.time()
        print(f'{fold} done, time: {time2-time1:.2f} s')

