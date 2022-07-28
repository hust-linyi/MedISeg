import sys
sys.path.append('../')
import os
import numpy as np
import NetworkTrainer.utils.util as utils
from NetworkTrainer.options import Options
import pandas as pd
from NetworkTrainer.utils.test_util import get_imglist, calculate_metric_percase
from tqdm import tqdm
import glob

def get_ensemble(mode='vote'):
    opt = Options(isTrain=False)
    opt.parse()
    save_dir = os.path.join(opt.result_dir, opt.task)
    fold_list = [0, 1, 2, 3, 4]
    metric_names = ['recall1', 'precision1', 'dice1', 'miou1', 'recall2', 'precision2', 'dice2', 'miou2']
    all_result = utils.AverageMeterArray(len(metric_names))
    results_all = list()

    for fold in fold_list:
        result_dir = os.path.join(save_dir, f'fold_{fold}', 'all')
        image_list = [f.replace('_pred.npy', '') for f in os.listdir(result_dir) if f.endswith('_pred.npy')]
        for image_path in tqdm(image_list):
            gt = np.load(os.path.join(opt.root_dir, image_path + '_label.npy'))
            if mode == 'vote':
                pred = np.load(os.path.join(result_dir, image_path + '_pred.npy'))
            else:
                pred = np.load(os.path.join(result_dir, image_path + '_prob.npy'))
            metrics = calculate_metric_percase(pred, gt)
            print(metrics)
            all_result.update([metrics[metric_name] for metric_name in metric_names])
        results_all.append([f'{result*100}' for result in all_result.avg])
    results_all_np = np.array(results_all)
    results_all_df = pd.DataFrame(results_all_np, columns=metric_names)
    results_all_df.to_csv(os.path.join(save_dir, f'test_results_{mode}.csv'), index=False)


def get_ensemble_pred():
    opt = Options(isTrain=False)
    opt.parse()
    save_dir = os.path.join(opt.result_dir, opt.task)
    fold_list = [0, 1, 2, 3, 4]
    seed_list = [2022, 2023, 2024, 2025, 2026]

    for fold in fold_list:
        out_dir = os.path.join(save_dir, f'fold_{fold}', 'all')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        image_list = os.listdir(os.path.join(save_dir, f'fold_{fold}', f'{seed_list[0]}', 'test_results', 'img'))
        image_list = [f.replace('_pred.npy', '') for f in image_list if f.endswith('_pred.npy')]

        for image_name in image_list:
            pred, prob = [], []
            for i, seed in enumerate(seed_list):
                result_dir = os.path.join(save_dir, f'fold_{fold}', f'{seed}', 'test_results', 'img')
                _pred = np.load(os.path.join(result_dir, image_name + '_pred.npy'))
                _pred_onehot = np.eye(3)[_pred]
                _pred_onehot = np.moveaxis(_pred_onehot, -1, 0)
                pred.append(_pred_onehot)
                prob.append(np.load(os.path.join(result_dir, image_name + '_prob.npy')))
            pred = np.mean(pred, axis=0)
            prob = np.mean(prob, axis=0)
            pred = pred.argmax(axis=0)
            prob = prob.argmax(axis=0)
            np.save(os.path.join(out_dir, image_name + '_pred.npy'), pred)
            np.save(os.path.join(out_dir, image_name + '_prob.npy'), prob)
    print('finished')


if __name__=='__main__':
    get_ensemble_pred()
    get_ensemble('avg')
    get_ensemble('vote')
