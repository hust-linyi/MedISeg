import os
import numpy as np
import cv2
import argparse
import tqdm


def convert_isic(data_dir):
    """
    Convert ISIC data to npy files
    The data can be downloaded from https://challenge.isic-archive.com/data/#2018
    license: https://creativecommons.org/licenses/by-nc-sa/4.0/
    """
    # Path to the ISIC data
    image_dir = os.path.join(data_dir, 'ISIC2018_Task1-2_Training_Input')
    mask_dir = os.path.join(data_dir, 'ISIC2018_Task1_Training_GroundTruth')

    # Path to the npy files
    npy_dir = os.path.join(data_dir, 'NumpyData')

    # Create the npy directories if they don't exist
    if not os.path.exists(npy_dir):
        os.makedirs(npy_dir)

    # Get the image and mask file names
    image_names = os.listdir(image_dir)
    mask_names = os.listdir(mask_dir)

    # Sort the file names
    image_names.sort()
    mask_names.sort()
    # Create lists to store the image and mask npy file names
    image_name_list, image_npy_list, mask_npy_list = [], [], []
    # Convert the images and masks to npy files
    for mask_name in tqdm.tqdm(mask_names):
        # Read the image and mask
        if not mask_name.endswith('.png'):
            continue
        image_name = mask_name.replace('_segmentation.png', '.jpg')
        image = cv2.imread(os.path.join(image_dir, image_name))
        mask = cv2.imread(os.path.join(mask_dir, mask_name))

        # Resize the image and mask to 256x256
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256))

        image_name_list.append(image_name)
        image_npy_list.append(image)
        mask_npy_list.append(mask)
    
    # Save the image and mask npy files
    image_name_npy = np.array(image_name_list)
    image_npy = np.array(image_npy_list)
    mask_npy = np.array(mask_npy_list)
    np.save(os.path.join(npy_dir, 'filename.npy'), image_name_npy)
    np.save(os.path.join(npy_dir, 'img.npy'), image_npy)
    np.save(os.path.join(npy_dir, 'mask.npy'), mask_npy)
        
    print('ISIC data with in total {} imgs converted to npy files, saved in {}'.format(len(image_name_list) ,npy_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert data to npy files')
    parser.add_argument('--data_dir', type=str, default=os.path.expanduser("~") + '/DATA/ISIC2018/TASK1')
    args = parser.parse_args()
    # Convert the ISIC data
    convert_isic(data_dir=args.data_dir)