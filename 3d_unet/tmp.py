"""
check the npy data size
"""
import os
import numpy as np
from rich import print
import time
import shutil

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
            if file.endswith('.gz') or file.endswith('gt.npy') or file.endswith('img.npy'):
                os.remove(os.path.join(root, file))
            # elif file.endswith('.npy'):
            #     data = np.load(os.path.join(root, file))
            #     np.savez_compressed(os.path.join(root, file.replace('.npy', '.npz')), data)
            #     os.remove(os.path.join(root, file))

    size_after = check_size(path)
    print(f'{path}, BEFORE: {size_before/1024/1024/1024:.2f} GB, AFTER: {size_after/1024/1024/1024:.2f} GB')


def rm_pycache(path):
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            if dir == '__pycache__':
                shutil.rmtree(os.path.join(root, dir))


def rm_checkpoint(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('1.pth.tar'):
                os.remove(os.path.join(root, file))

if __name__ == '__main__':
    # data_dir = '/newdata/ianlin/Experiment/KIT19/kit19/'
    # fold_list = os.listdir(data_dir)

    # for fold in fold_list:
    #     time1 = time.time()
    #     com_dir = os.path.join(data_dir, fold)
    #     compress_npy(com_dir)
    #     time2 = time.time()
    #     print(f'{fold} done, time: {time2-time1:.2f} s')

    # data_dir = '/newdata/ianlin/CODE/seg_trick/'
    # rm_pycache(data_dir)

    data_dir = '/newdata/ianlin/Experiment/ISIC-2018/isic2018'
    rm_checkpoint(data_dir)
