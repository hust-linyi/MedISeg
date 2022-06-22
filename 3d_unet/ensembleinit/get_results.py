from fileinput import filename
import os
import numpy as np
import utils.util as utils
from utils.accuracy import compute_metrics
from options import Options
import cv2
import pandas as pd
from utils.test_util import get_imglist, calculate_metric_percase
from tqdm import tqdm

def get_ensemble(mode='vote'):
    opt = Options(isTrain=False)
    opt.parse()
    save_dir = os.path.join(opt.result_dir, opt.task, opt.model['name'])
    fold_list = [0, 1, 2, 3, 4]
    seed_list = [2022, 2023, 2024, 2025, 2026]
    metric_names = ['p_recall', 'p_precision', 'dice', 'miou']
    all_result = utils.AverageMeter(len(metric_names))
    results_all = list()
  

    for fold in fold_list:
        image_list = get_imglist(opt.root_dir, fold)
        for image_path in tqdm(image_list):
            gt = np.load(image_path.replace('img', 'gt'))
            filename = os.path.basename(image_path).replace('_img.npy', '')
            pred = 0
            for i, seed in enumerate(seed_list):
                result_dir = os.path.join(save_dir, f'fold_{fold}', f'{seed}', 'test_results', 'img')
                if mode == 'vote':
                    pred += np.load(os.path.join(result_dir, filename + '_pred.png'))
                else:
                    pred += np.load(os.path.join(result_dir, filename + '_prob.npy'))
            pred /= len(seed_list)
            pred = pred > 0.5
            metrics = compute_metrics(pred, gt, metric_names)
            all_result.update([metrics[metric_name] for metric_name in metric_names])

        results_all.append([f'{result*100}' for result in all_result.avg])
    results_all_np = np.array(results_all)
    results_all_df = pd.DataFrame(results_all_np, columns=metric_names)
    results_all_df.to_csv(os.path.join(save_dir, f'test_results_{mode}.csv'), index=False)


if __name__=='__main__':
    # get_ensemble('avg')
    get_ensemble('vote')