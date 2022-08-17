import sys
sys.path.append('../')
import os
import numpy as np
from NetworkTrainer.utils.util import AverageMeterArray
from NetworkTrainer.utils.test_util import test_all_case, calculate_metric_percase
from NetworkTrainer.options.options import Options
import cv2
import pandas as pd


def load_data(data_dir):
    if 'png' in data_dir:
        # load prediction
        out = cv2.imread(data_dir, cv2.IMREAD_GRAYSCALE)
        out = out / 255.0
    elif 'npy' in data_dir: # mode == 'avg'
        # load probability
        out = np.load(data_dir)
    else:
        raise ValueError('data_dir must be either png or npy')
    return out

def get_ensemble(mode='vote'):
    opt = Options(isTrain=False)
    opt.parse()
    save_dir = os.path.join(opt.result_dir, 'ensembleinit')
    fold_list = [0, 1, 2, 3, 4]
    seed_list = [2022, 2023, 2024, 2025, 2026]
    metric_names = ['recall1', 'precision1', 'dice1', 'miou1']
    total_metric = AverageMeterArray(len(metric_names))
    results_all = list()

    for fold in fold_list:
        result_dir = os.path.join(save_dir, f'fold_{fold}', f'{seed_list[0]}', 'test_results', 'img')
        filenames = [f.replace('_pred.npy', '') for f in os.listdir(result_dir) if f.endswith('_pred.npy')]
        for filename in filenames:
            # gt = load_data(os.path.join(result_dir, filename + '_gt.png'))
            gt = load_data(os.path.join(opt.root_dir, filename + '_label.npy'))
            pred = 0
            for i, seed in enumerate(seed_list):
                result_dir = os.path.join(save_dir, f'fold_{fold}', f'{seed}', 'test_results', 'img')
                if mode == 'vote':
                    pred += load_data(os.path.join(result_dir, filename + '_pred.npy'))
                else:
                    pred += load_data(os.path.join(result_dir, filename + '_prob.npy'))[1]

            # import ipdb;ipdb.set_trace()
            pred = pred.astype(np.float32) / len(seed_list)
            pred = pred > 0.5
            single_metric = calculate_metric_percase(pred, gt)
            total_metric.update([single_metric[metric_name] for metric_name in metric_names])
            print(single_metric)

        results_all.append([f'{result*100}' for result in total_metric.avg])
    results_all_np = np.array(results_all)
    results_all_df = pd.DataFrame(results_all_np, columns=metric_names)
    results_all_df.to_csv(os.path.join(save_dir, f'test_results_{mode}.csv'), index=False)


if __name__=='__main__':
    get_ensemble('avg')
    # get_ensemble('vote')