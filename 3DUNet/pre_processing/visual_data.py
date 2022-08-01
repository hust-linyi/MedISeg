import os
from PIL import Image
import numpy as np
import shutil
import pickle
from tqdm import tqdm

def rm_n_mkdir(dir_path):
    """Remove and then make a new directory."""
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

def save_png(img, save_dir):
    rm_n_mkdir(save_dir)
    for i in range(0, img.shape[0], 5):
        img_ = img[i, :, :]
        img_ = (img_ - img_.min()) / (img_.max() - img_.min())
        img_ = (img_ * 255).astype(np.uint8)
        img_ = Image.fromarray(img_).convert("L")
        img_.save(os.path.join(save_dir, '%d.jpg' % i))
    

def check_data():
    # img_dir = '/newdata/ianlin/Data/COVID-19-20/monai/preprocess'
    # save_dir = '/mnt/yfs/ianlin/Experiment/COVID/visual'
    img_dir = '/home/ylindq/Data/LIVER/monai/raw/imagesTr'
    save_dir = '/home/ylindq/Experiment/LIVER/visual'
    filenames = [f for f in os.listdir(img_dir) if f.endswith('image.npy')]
    filenames.sort()
    for filename in tqdm(filenames):
        img = np.load(os.path.join(img_dir, filename))
        save_png(img, os.path.join(save_dir, filename.replace('image.npy', '')))


def check_data_pro():
    # pkl_pth = '/newdata/ianlin/Data/COVID-19-20/monai/preprocess/dataset_pro.pkl'
    # data_dir = '/mnt/yfs/ianlin/Data/COVID-19-20/COVID-19-20_v2/preprocess/monai/raw/imagesTr'
    pkl_pth = '/home/ylindq/Data/LIVER/monai/preprocess/dataset_pro.pkl'
    data_dir = '/home/ylindq/Data/LIVER/monai/raw/imagesTr'
    data_info = pickle.load(open(pkl_pth, 'rb'))
    for patient_id in data_info['patient_names']:
        # print(data_info['dataset_properties'][patient_id]['origin'])
        print(data_info['dataset_properties'][patient_id]['spacing'])
        # print(data_info['dataset_properties'][patient_id]['direction'])
        # print(data_info['dataset_properties'][patient_id]['size'])

        # if data_info['dataset_properties'][patient_id]['direction'][-1] == -1:
        #     # reverse the z direction
        #     img = np.load(os.path.join(data_dir, patient_id + '_image.npy'))
        #     label = np.load(os.path.join(data_dir, patient_id + '_label.npy'))
        #     img = np.flip(img, axis=-1)
        #     label = np.flip(label, axis=-1)
        #     np.save(os.path.join(data_dir, patient_id + '_image.npy'), img)
        #     np.save(os.path.join(data_dir, patient_id + '_label.npy'), label)



if __name__=='__main__':
    check_data()
    # check_data_pro()