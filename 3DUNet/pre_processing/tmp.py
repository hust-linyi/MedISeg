import os
import numpy as np

data_dir = '/newdata/ianlin/Data/LIVER/lits2017/preprocess'
filenames = [f for f in os.listdir(data_dir) if 'label' in f]

for f in filenames:
    img = np.load(os.path.join(data_dir, f.replace('label', 'image')))
    label = np.load(os.path.join(data_dir, f))
    if img.shape != label.shape:
        print(f)
        print(img.shape)
        print(label.shape)
