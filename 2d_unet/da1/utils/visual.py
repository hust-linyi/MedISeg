import cv2
import numpy as np
import utils
from options import Options
import os
from test import data_split
from skimage import measure
from utils.evaluate import rm_n_mkdir
from utils.accuracy import compute_metrics


def draw_boundary(img, label, set_color=None):
    img_bnd = img.copy()
    label = measure.label(label)
    for i in range(1, label.max()+1):
        if set_color is None:
            color = np.array(utils.get_random_color())
            color = (color * 255).astype(np.uint8)
            color = (int(color[0]), int(color[1]), int(color[2])) 
        else:
            color = set_color
        mask = label == i
        countours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img_bnd, countours, -1, color, 2)
    return img_bnd


if __name__=='__main__':
    metric_names = ['aji']
    fold = 0
    crop_size = 250
    save_num = 20
    data_dir = '/home/ylindq/Data/cervical_cell_original/np/'
    out_dir = './tmp/visual/'
    pred_dir_nuc_baseline = f'/home/ylindq/Experiment/cps/nuc/baseline/res34/fold_{fold}/test_results'
    pred_dir_cyt_baseline = f'/home/ylindq/Experiment/cps/cyt/baseline/res34/fold_{fold}/test_results'
    pred_dir_nuc_insmix = f'/home/ylindq/Experiment/cps/nuc/insmix/res34/fold_{fold}/test_results'
    pred_dir_cyt_insmix = f'/home/ylindq/Experiment/cps/cyt/insmix/res34/fold_{fold}/test_results'

    imgs = np.load(os.path.join(data_dir, 'nuc', 'data_after_stain_norm_ref1.npy'))
    gts_nuc = np.load(os.path.join(data_dir, 'nuc', 'ist.npy'))
    gts_cyt = np.load(os.path.join(data_dir, 'cyt', 'ist.npy'))
    imgs = data_split(imgs, fold)
    gts_nuc = data_split(gts_nuc, fold)
    gts_cyt = data_split(gts_cyt, fold)
    rm_n_mkdir(out_dir)

    score_list_nuc, score_list_cyt = [], []
    save_list = []
    for idx in range(imgs.shape[0]):
    # for idx in range(1):
        img = imgs[idx]
        gt_nuc = gts_nuc[idx]
        gt_cyt = gts_cyt[idx]
        pred_nuc_baseline = cv2.imread(os.path.join(pred_dir_nuc_baseline, '{:d}_colored_seg.png'.format(idx)), cv2.IMREAD_GRAYSCALE)
        pred_cyt_baseline = cv2.imread(os.path.join(pred_dir_cyt_baseline, '{:d}_colored_seg.png'.format(idx)), cv2.IMREAD_GRAYSCALE)
        pred_nuc_insmix = cv2.imread(os.path.join(pred_dir_nuc_insmix, '{:d}_colored_seg.png'.format(idx)), cv2.IMREAD_GRAYSCALE)
        pred_cyt_insmix = cv2.imread(os.path.join(pred_dir_cyt_insmix, '{:d}_colored_seg.png'.format(idx)), cv2.IMREAD_GRAYSCALE)

        for i in range(img.shape[0]//crop_size):
            for j in range(img.shape[1]//crop_size):
                _img = img[i*crop_size:(i+1)*crop_size, j*crop_size:(j+1)*crop_size]
                _gt_nuc = gt_nuc[i*crop_size:(i+1)*crop_size, j*crop_size:(j+1)*crop_size]
                _gt_cyt = gt_cyt[i*crop_size:(i+1)*crop_size, j*crop_size:(j+1)*crop_size]
                _pred_nuc_baseline = pred_nuc_baseline[i*crop_size:(i+1)*crop_size, j*crop_size:(j+1)*crop_size]
                _pred_cyt_baseline = pred_cyt_baseline[i*crop_size:(i+1)*crop_size, j*crop_size:(j+1)*crop_size]
                _pred_nuc_insmix = pred_nuc_insmix[i*crop_size:(i+1)*crop_size, j*crop_size:(j+1)*crop_size]
                _pred_cyt_insmix = pred_cyt_insmix[i*crop_size:(i+1)*crop_size, j*crop_size:(j+1)*crop_size]

                try:
                    aji_nuc_baseline = compute_metrics(_pred_nuc_baseline, _gt_nuc, metric_names)
                    aji_cyt_baseline = compute_metrics(_pred_cyt_baseline, _gt_cyt, metric_names)
                except:
                    aji_nuc_baseline = {'aji': 0}
                    aji_cyt_baseline = {'aji': 0}
                try:
                    aji_nuc_insmix = compute_metrics(_pred_nuc_insmix, _gt_nuc, metric_names)
                    aji_cyt_insmix = compute_metrics(_pred_cyt_insmix, _gt_cyt, metric_names)
                except:
                    aji_nuc_insmix = {'aji': 0}
                    aji_cyt_insmix = {'aji': 0}
                score_list_nuc.append(aji_nuc_insmix['aji']-aji_nuc_baseline['aji'])
                score_list_cyt.append(aji_cyt_insmix['aji']-aji_cyt_baseline['aji'])
                save_list.append((_img, _gt_nuc, _gt_cyt, _pred_nuc_baseline, _pred_cyt_baseline, _pred_nuc_insmix, _pred_cyt_insmix))

    score_list_sort_nuc = score_list_nuc.copy()
    score_list_sort_cyt = score_list_cyt.copy()
    score_list_sort_nuc.sort(reverse = True)
    score_list_sort_cyt.sort(reverse = True)

    for i in range(save_num):
        s_nuc = score_list_sort_nuc[i]
        s_cyt = score_list_sort_cyt[i]
        img_n, g_n, _, p_n_b, _, p_n_i, _ = save_list[score_list_nuc.index(s_nuc)]
        img_c, _, g_c, _, p_c_b, _, p_c_i = save_list[score_list_cyt.index(s_cyt)]

        # draw boudary
        img_bnd_n_g = draw_boundary(img_n, g_n, (255, 0, 0))
        img_bnd_n_b = draw_boundary(img_n, p_n_b, (255, 0, 0))
        img_bnd_n_i = draw_boundary(img_n, p_n_i, (255, 0, 0))
        img_bnd_c_g = draw_boundary(img_c, g_c)
        img_bnd_c_b = draw_boundary(img_c, p_c_b)
        img_bnd_c_i = draw_boundary(img_c, p_c_i)

        cv2.imwrite(f'{out_dir}/{i}_nuc_gt.png', img_bnd_n_g[..., ::-1])
        cv2.imwrite(f'{out_dir}/{i}_nuc_baseline.png', img_bnd_n_b[..., ::-1])
        cv2.imwrite(f'{out_dir}/{i}_nuc_insmix.png', img_bnd_n_i[..., ::-1])

        cv2.imwrite(f'{out_dir}/{i}_cyt_gt.png', img_bnd_c_g[..., ::-1])
        cv2.imwrite(f'{out_dir}/{i}_cyt_baseline.png', img_bnd_c_b[..., ::-1])
        cv2.imwrite(f'{out_dir}/{i}_cyt_insmix.png', img_bnd_c_i[..., ::-1])

 