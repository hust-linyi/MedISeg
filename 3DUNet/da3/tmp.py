import os
import numpy as np

def check_data(data_path):
    files = [f for f in os.listdir(data_path) if f.endswith('.npy')]
    for f in files:
        data = np.load(os.path.join(data_path, f))
        if np.isnan(data).any():
            print(f'{f} has nan')
            os.remove(os.path.join(data_path, f))


if __name__=='__main__':
    data_dir = '/newdata/ianlin/Data/KIT-19/aug'
    check_data(data_dir)