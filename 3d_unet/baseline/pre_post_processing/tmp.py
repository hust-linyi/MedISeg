import os
import pickle

data_dir = '/mnt/yfs/ianlin/Data/COVID-19-20/COVID-19-20_v2/preprocess/monai/raw'
data_info = pickle.load(open(os.path.join(data_dir, 'dataset_pro.pkl'), 'rb'))
print(data_info)
