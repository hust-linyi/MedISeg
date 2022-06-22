"""
check the npy data size
"""
import os
import numpy as np

def get_size(path):
    size = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            size = os.path.getsize(os.path.join(root, file))
            print(f'{file}, {size/1024/1024} MB')


root_dir = '/newdata/ianlin/Experiment/KIT19/kit19/baseline/fold_0/test_results/img'
out_dir = '/newdata/ianlin/Experiment/tmp'
case_name = 'case_00025'

img = np.load(os.path.join(root_dir, case_name + '_img.npy'))
gt = np.load(os.path.join(root_dir, case_name + '_gt.npy'))
pred = np.load(os.path.join(root_dir, case_name + '_pred.npy'))
prob = np.load(os.path.join(root_dir, case_name + '_prob.npy'))

img_int = img.astype(int)
gt_int = gt.astype(int)
pred_int = pred.astype(int)
prob_int = prob.astype(int)

np.save(os.path.join(out_dir, case_name + '_img_int.npy'), img_int)
np.save(os.path.join(out_dir, case_name + '_gt_int.npy'), gt_int)
np.save(os.path.join(out_dir, case_name + '_pred_int.npy'), pred_int)
np.save(os.path.join(out_dir, case_name + '_prob_int.npy'), prob_int)

np.save(os.path.join(out_dir, case_name + '_img.npy'), img)
np.save(os.path.join(out_dir, case_name + '_gt.npy'), gt)
np.save(os.path.join(out_dir, case_name + '_pred.npy'), pred)
np.save(os.path.join(out_dir, case_name + '_prob.npy'), prob)

get_size(out_dir)

