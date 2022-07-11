import os
import pandas as pd
import numpy as np
from rich import print

def get_results(save_dir):
    fold_list = [0, 1, 2, 3, 4]
    metric_names = ['p_recall', 'p_precision', 'dice', 'miou']
    results_all = list()
    for fold in fold_list:
        result_dir = save_dir.replace('fold_0', f'fold_{fold}')
        result = pd.read_csv(os.path.join(result_dir, 'test_results.csv')).values
        results_all.append(result)
    results_all = np.concatenate(results_all, axis=0)
    results_mean = np.mean(results_all, axis=0)
    results_all = np.concatenate([results_all, results_mean.reshape(1,4)], axis=0)
    results_all = pd.DataFrame(results_all, columns=metric_names)
    results_all.to_csv(os.path.join('/'.join(result_dir.split('/')[:-2]), 'test_results_all.csv'), index=False)

def if_complete_train(path):
    flag = True
    if not os.path.exists(path):
        flag = False
    else:
        for i in range(5):
            if not os.path.exists(os.path.join(path.replace('fold_0', f'fold_{i}'), 'test_results.csv')):
                flag = False
    return flag

if __name__ == '__main__':
    # data_dir = '/newdata/ianlin/Experiment/ISIC-2018/isic2018'
    data_dir = '/newdata/ianlin/Experiment/CoNIC_Challenge/conic'
    black_list = ['DEBUG']
    folds = [f for f in os.listdir(data_dir) if f not in black_list]
    folds.sort()
    for fold in folds:
        save_dir = os.path.join(data_dir, fold, 'res50', 'fold_0', 'test_results')
        if fold == 'pt1k':
            save_dir = os.path.join(data_dir, fold, 'res50_1k', 'fold_0', 'test_results')
        elif fold == 'pt21k':
            save_dir = os.path.join(data_dir, fold, 'res50_21k', 'fold_0', 'test_results')
        if if_complete_train(save_dir):
            get_results(save_dir)
            print(f'{fold}')
            final_results_path = '/'.join(save_dir.split('/')[:-2]) + '/test_results_all.csv'
            final_results = pd.read_csv(final_results_path, index_col=0)
            print(final_results)
