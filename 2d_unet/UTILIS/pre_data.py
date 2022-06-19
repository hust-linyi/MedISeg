"""
This is the preprocess the dataset as follows:
1. Resize the image to the [256, 192]
2. save the imgs to a single npy file;
3. save the labels to a single npy file;
"""

import os
import cv2
import numpy as np
import albumentations as A

data_dir = '/home/ylindq/Data/ISIC-2018/'
img_dir = data_dir + 'IMG'
mask_dir = data_dir + 'MASK'
files = [os.path.splitext(file)[0] for file in os.listdir(img_dir) if file.endswith('.jpg')]
resize = A.Resize(192, 256)

img_list, mask_list = [], []
for file in files:
    img = cv2.imread(os.path.join(img_dir, file + '.jpg'))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(os.path.join(mask_dir, file + '_segmentation.png'), cv2.IMREAD_GRAYSCALE)
    mask = mask / 255.0
    transformed = resize(image=img, mask=mask)
    img, mask = transformed['image'], transformed['mask']
    img_list.append(img)
    mask_list.append(mask)

img_np = np.array(img_list)
mask_np = np.array(mask_list)
np.save(os.path.join(data_dir, 'NumpyData','img.npy'), img_np)
np.save(os.path.join(data_dir, 'NumpyData','mask.npy'), mask_np)
np.save(os.path.join(data_dir, 'NumpyData','filename.npy'), files)




