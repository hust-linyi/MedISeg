# import cv2
import numpy as np
import os

def save_to_png(img, path):
    # cv2.imwrite(path, img)
    return


if __name__=='__main__':
    data_dir = '/newdata/ianlin/Data/KIT-19/aug'
    filenames = [f.replace('_image.npy', '') for f in os.listdir(data_dir) if f.endswith('_image.npy')]
    for f in filenames:
        img = np.load(os.path.join(data_dir, f+'_image.npy'))
        label = np.load(os.path.join(data_dir, f+'_label.npy'))
        # print(f'==> {f}, {img.shape}, {label.shape}, {np.max(img)}, {np.max(label)}, {np.min(img)}, {np.min(label)}')
        if np.isnan(np.max(img)) or np.isnan(np.min(img)) or np.isnan(np.max(label)) or np.isnan(np.min(label)):
            print(f'==> {f}, {img.shape}, {label.shape}, {np.max(img)}, {np.max(label)}, {np.min(img)}, {np.min(label)}')
