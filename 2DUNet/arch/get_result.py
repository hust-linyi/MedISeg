import sys
sys.path.append('../')
import os
import pandas as pd
import numpy as np
from rich import print

def get_results(result_dir):
    fold_list = [0, 1, 2, 3, 4]
    metric_names = ['p_recall', 'p_precision', 'dice', 'miou']
    results_all = list()
    for fold in fold_list:
        result = pd.read_csv(os.path.join(result_dir, f'fold_{fold}', 'test_results', 'test_results.csv')).values
        results_all.append(result)
    results_all = np.concatenate(results_all, axis=0)
    results_mean = np.mean(results_all, axis=0)
    results_all = np.concatenate([results_all, results_mean.reshape(1,4)], axis=0)
    results_all = pd.DataFrame(results_all, columns=metric_names)
    results_all.to_csv(os.path.join(result_dir, 'test_results_all.csv'), index=False)
    print(result_dir.split('/')[-1])
    print(results_all)

def if_complete_train(path):
    flag = True
    if not os.path.exists(path):
        flag = False
    else:
        for i in range(5):
            if not os.path.exists(os.path.join(path, f'fold_{i}', 'test_results', 'test_results.csv')):
                flag = False
    return flag


if __name__ == '__main__':
    # data_dir = '/newdata/ianlin/Experiment/ISIC-2018/isic2018/arch'
    data_dir = '/newdata/ianlin/Experiment/CoNIC_Challenge/conic/arch'
    folds = os.listdir(data_dir)
    folds.sort()
    for fold in folds:
        result_dir = os.path.join(data_dir, fold)
        if if_complete_train(result_dir):
            get_results(result_dir)

