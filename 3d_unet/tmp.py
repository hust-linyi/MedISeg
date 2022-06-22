"""
check the npy data size
"""
import os
import numpy as np
from rich import print

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
            elif file.endswith('.npy'):
                print(f'{file}')
                data = np.load(os.path.join(root, file))
                np.savez_compressed(os.path.join(root, file.replace('.npy', '.npz')), data)
                os.remove(os.path.join(root, file))

    size_after = check_size(path)
    print(f'{path}, BEFORE: {size_before/1024/1024/1024:.2f} GB, AFTER: {size_after/1024/1024/1024:.2f} GB')


if __name__ == '__main__':
    data_dir = '/newdata/ianlin/Experiment/KIT19/kit19/patch32'
    compress_npy(data_dir)

