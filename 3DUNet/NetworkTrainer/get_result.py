import sys
sys.path.append('../')
import os
import pandas as pd
from options import Options
import numpy as np

def get_results():
    opt = Options(isTrain=False)
    opt.parse()

    fold_list = [0, 1, 2, 3, 4]
    metric_names = ['recall1', 'precision1', 'dice1', 'miou1', 'recall2', 'precision2', 'dice2', 'miou2']
    results_all = list()
    print(opt.test['save_dir'])
    for fold in fold_list:
        result_dir = opt.test['save_dir'].replace('fold_0', f'fold_{fold}')
        result = pd.read_csv(os.path.join(result_dir, 'test_results.csv')).values
        results_all.append(result)
    results_all = np.concatenate(results_all, axis=0)
    results_mean = np.mean(results_all, axis=0)
    results_all = np.concatenate([results_all, results_mean.reshape(1,8)], axis=0)
    results_all = pd.DataFrame(results_all, columns=metric_names)
    results_all.to_csv(os.path.join('/'.join(result_dir.split('/')[:-2]), 'test_results_all.csv'), index=False)
    final_results = pd.read_csv(os.path.join('/'.join(result_dir.split('/')[:-2]), 'test_results_all.csv'), index_col=0)
    print(final_results)

if __name__ == '__main__':
    get_results()

