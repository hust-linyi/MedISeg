
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import skimage.morphology as morph
import scipy.ndimage.morphology as ndi_morph
from skimage import measure
from scipy import misc
from models.modelU import ResUNet34, ResUNet
# from model_UNet import UNet
# from DenseUnet import UNet
from models.model_UNet import UNet
import utils.utils as utils
from utils.accuracy import compute_metrics
import time
import imageio
from options import Options
from utils.my_transforms import get_transforms
from rich import print
from tqdm import tqdm
from utils.dataset import DataFolder
from torch.utils.data import DataLoader
from scipy import ndimage
from multiprocessing import Pool
import cv2


def main():
    opt = Options(isTrain=False)
    opt.parse()
    opt.save_options()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.test['gpus'])

    save_dir = opt.test['save_dir']
    model_path = opt.test['model_path']
    save_flag = opt.test['save_flag']
    # img_dir = opt.test['img_dir']
    # label_dir = opt.test['label_dir']
    # img_path = os.path.join(opt.root_dir, 'test', 'data_after_stain_norm_ref1.npy')
    # gt_path = os.path.join(opt.root_dir, 'test', 'gt.npy')
    # img_path = os.path.join(opt.root_dir, 'data_after_stain_norm_ref1.npy')
    # gt_path = os.path.join(opt.root_dir, 'gt.npy')
    # bnd_path = os.path.join(opt.root_dir, 'bnd.npy')

    test_set = DataFolder(root_dir=opt.root_dir, phase='test', data_transform=opt.transform['test'], fold=opt.fold)
    test_loader = DataLoader(test_set, batch_size=opt.test['batch_size'], shuffle=False, drop_last=False)

    # data transforms
    # test_transform = get_transforms(opt.transform['test'])
    # test_set = DataFolder(root_dir=opt.root_dir, phase='val', data_transform=test_transform)
    # test_loader = DataLoader(test_set, batch_size=opt.test['batch_size'], shuffle=False, drop_last=False)
    
    if 'res' in opt.model['name']:
        model = ResUNet(net=opt.model['name'], seg_classes=2, colour_classes=3, pretrained=opt.model['pretrained'])   
    else:
        model = UNet(3, 2, 2)    
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    cudnn.benchmark = True

    # ----- load trained model ----- #
    print(f"=> loading trained model in {model_path}")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    print("=> loaded model at epoch {}".format(checkpoint['epoch']))
    model = model.module

    # switch to evaluate mode
    model.eval()

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if save_flag:
        if not os.path.exists(os.path.join(save_dir, 'img')):
            os.mkdir(os.path.join(save_dir, 'img'))

    # metric_names = ['acc', 'p_F1', 'p_recall', 'p_precision', 'dice', 'aji']
    metric_names = ['acc', 'p_F1', 'p_recall', 'p_precision', 'dice', 'miou']
    test_results = dict()
    all_result = utils.AverageMeter(len(metric_names))

    for i, (input, gt, name) in enumerate(tqdm(test_loader)):
        input = input.cuda()

        output = model(input)
        pred = output.data.max(1)[1].cpu().numpy()

        for j in range(pred.shape[0]):
            metrics = compute_metrics(pred[j], gt[j], metric_names)
            all_result.update([metrics[metric_name] for metric_name in metric_names])
            if save_flag:
                imageio.imwrite(os.path.join(save_dir, 'img', f'{name[j]}_pred.png'), (pred[j] * 255).astype(np.uint8))
                imageio.imwrite(os.path.join(save_dir, 'img', f'{name[j]}_gt.png'), (gt[j].numpy() * 255).astype(np.uint8))

    for i in range(len(metric_names)):
        print(f"{metric_names[i]}: {all_result.avg[i]:.4f}", end='\t')

    header = metric_names
    utils.save_results(header, all_result.avg, test_results, '{:s}/test_results_epoch_{:d}_AJI_{:.4f}.txt'.format(save_dir, checkpoint['epoch'], all_result.avg[5]))



#     imgs = np.load(img_path)
#     gts = np.load(gt_path)
#     bnds = np.load(bnd_path)
#     imgs = data_split(imgs, opt.fold)
#     gts = data_split(gts, opt.fold)
#     bnds = data_split(bnds, opt.fold)
#     with torch.no_grad():
#         img_process = tqdm(range(imgs.shape[0]))
#         for idx in img_process:
#             # load test image
#             img_process.set_description_str('=> Processing image {:d}'.format(idx))
#             # print('=> Processing image {:s}'.format(img_name))
#             # img_path = '{:s}/{:s}'.format(img_dir, img_name)
#             # img = Image.open(img_path)
#             img = Image.fromarray(imgs[idx])

#             ori_h = img.size[1]
#             ori_w = img.size[0]
#             # name = os.path.splitext(img_name)[0]
#             name = str(idx)
#             # label_path = '{:s}/{:s}_label.png'.format(label_dir, name)
#             # gt = imageio.imread(label_path)
#             gt = gts[idx]
#             bnd = bnds[idx]
        
#             input = test_transform((img,))[0].unsqueeze(0)
#             prob_maps_in, prob_maps_bnd = get_probmaps(input, model, opt)
#             prob_in, prob_bnd = prob_maps_in[1], prob_maps_bnd[1]
#             pred_labeled = post_proc(prob_in-prob_bnd, post_dilation_iter=2).squeeze()
#             pred_in_labeled = post_proc(prob_in, post_dilation_iter=2).squeeze()

#             pred_in = np.argmax(prob_maps_in, axis=0)  # prediction
#             pred_bnd = np.argmax(prob_maps_bnd, axis=0)  # prediction

#             # pred_labeled = measure.label(pred_in)
#             # pred_labeled = morph.remove_small_objects(pred_labeled, opt.post['min_area'])
#             # pred_labeled = ndi_morph.binary_fill_holes(pred_labeled > 0)
#             # pred_labeled = measure.label(pred_labeled)

#             # print('\tComputing metrics...')
#             # metrics = compute_metrics(pred_labeled, gt, metric_names)

#             gt_bnd = ((gt - bnd) > 0).astype(np.uint8)
#             # metrics = compute_metrics(pred_labeled, gt, metric_names)
#             metrics = compute_metrics(pred_in_labeled, gt, metric_names)

#             # save result for each image
#             test_results[name] = [metrics['acc'], metrics['p_F1'], metrics['p_recall'], metrics['p_precision'],
#                                 metrics['dice'], metrics['aji']]
#             #
#             # # update the average result
#             all_result.update([metrics['acc'], metrics['p_F1'], metrics['p_recall'], metrics['p_precision'],
#                             metrics['dice'], metrics['aji']])
            
#             # try detect the boundary
#             # _img = np.array(img)
#             # _img = cv2.cvtColor(_img, cv2.COLOR_RGB2GRAY)
#             # edge_sobel, edge_scharr, edge_laplacian, edge_canny = utils.edge_detect(_img)
#             # cv2.imwrite('{:s}/{:s}_pred_bnd_sobel.png'.format(save_dir, name), (edge_sobel * 255).astype(np.uint8))
#             # cv2.imwrite('{:s}/{:s}_pred_bnd_scharr.png'.format(save_dir, name), (edge_scharr * 255).astype(np.uint8))
#             # cv2.imwrite('{:s}/{:s}_pred_bnd_laplacian.png'.format(save_dir, name), (edge_laplacian * 255).astype(np.uint8))
#             # cv2.imwrite('{:s}/{:s}_pred_bnd_canny.png'.format(save_dir, name), (edge_canny * 255).astype(np.uint8))

#             # # do some open operation
#             # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
#             # pred_in_bnd = (pred_in - pred_bnd).astype(np.uint8)
#             # erode1 = cv2.erode(pred_in_bnd, kernel, iterations=1)
#             # erode3 = cv2.erode(pred_in_bnd, kernel, iterations=3)
#             # erode10 = cv2.erode(pred_in_bnd, kernel, iterations=10)
#             # cv2.imwrite('{:s}/{:s}_pred_in_erod1.png'.format(save_dir, name), (erode1 * 255).astype(np.uint8))
#             # cv2.imwrite('{:s}/{:s}_pred_in_erod3.png'.format(save_dir, name), (erode3 * 255).astype(np.uint8))
#             # cv2.imwrite('{:s}/{:s}_pred_in_erod10.png'.format(save_dir, name), (erode10 * 255).astype(np.uint8))

#             # save gt image
#             gt_labeled = measure.label(gt)
#             gt_bnd_labeled = measure.label(gt_bnd)
#             gt_colored_instance = np.zeros((ori_h, ori_w, 3))
#             gt_bnd_colored_instance = np.zeros((ori_h, ori_w, 3))
#             for k in range(1, np.max(gt_labeled) + 1):
#                 gt_colored_instance[gt_labeled == k, :] = np.array(utils.get_random_color())
#             for k in range(1, np.max(gt_bnd_labeled) + 1):
#                 gt_bnd_colored_instance[gt_bnd_labeled == k, :] = np.array(utils.get_random_color())
#             imageio.imsave('{:s}/{:s}_colored_gt.png'.format(save_dir, name), (gt_colored_instance * 255).astype(np.uint8))
#             imageio.imsave('{:s}/{:s}_colored_gt_bnd.png'.format(save_dir, name), (gt_bnd_colored_instance * 255).astype(np.uint8))

#             # save image
#             if save_flag:
#                 # print('\tSaving image results...')
#                 imageio.imsave('{:s}/{:s}_pred_in.png'.format(save_dir, name), (pred_in * 255).astype(np.uint8))
#                 imageio.imsave('{:s}/{:s}_pred_bnd.png'.format(save_dir, name), (pred_bnd * 255).astype(np.uint8))
#                 # final_pred = Image.fromarray(pred_labeled.astype(np.uint16))
#                 # final_pred.save('{:s}/{:s}_seg.tiff'.format(seg_folder, name))

#                 # save colored objects
#                 pred_colored_instance = np.zeros((ori_h, ori_w, 3))
#                 for k in range(1, pred_labeled.max() + 1):
#                     pred_colored_instance[pred_labeled == k, :] = np.array(utils.get_random_color())
#                 filename = '{:s}/{:s}_colored_seg.png'.format(save_dir, name)
#                 imageio.imsave(filename, (pred_colored_instance * 255).astype(np.uint8))
#         # print('Average Acc: {r[0]:.4f}\tF1: {r[1]:.4f}\tDice: {r[4]:.4f}\tAJI: {r[5]:.4f}\n'.format(r=all_result.avg))
#         print('Average Acc: {r[0]:.4f}\tRecall: {r[2]:.4f}\tPrecision: {r[3]:.4f}\tDice: {r[4]:.4f}\tAJI: {r[5]:.4f}\n'.format(r=all_result.avg))

#         header = metric_names
#         utils.save_results(header, all_result.avg, test_results, '{:s}/test_results_epoch_{:d}_AJI_{:.4f}.txt'.format(save_dir, checkpoint['epoch'], all_result.avg[5]))


# def get_probmaps(input, model1, opt):
#     size = opt.test['patch_size']
#     overlap = opt.test['overlap']
#     output_in, output_bnd = split_forward(model1, input, size, overlap)

#     output_in = output_in.squeeze(0)
#     output_bnd = output_bnd.squeeze(0)
#     prob_maps1 = F.softmax(output_in, dim=0).cpu().numpy()
#     prob_maps2 = F.softmax(output_bnd, dim=0).cpu().numpy()
#     return prob_maps1, prob_maps2


# def split_forward(model, input, size, overlap, outchannel = 2):
#     '''
#     split the input image for forward passes
#     '''

#     b, c, h0, w0 = input.size()

#     # zero pad for border patches
#     pad_h = 0
#     if h0 - size > 0 and (h0 - size) % (size - overlap) > 0:
#         pad_h = (size - overlap) - (h0 - size) % (size - overlap)
#         tmp = torch.zeros((b, c, pad_h, w0))
#         input = torch.cat((input, tmp), dim=2)

#     if w0 - size > 0 and (w0 - size) % (size - overlap) > 0:
#         pad_w = (size - overlap) - (w0 - size) % (size - overlap)
#         tmp = torch.zeros((b, c, h0 + pad_h, pad_w))
#         input = torch.cat((input, tmp), dim=3)

#     _, c, h, w = input.size()

#     output1 = torch.zeros((input.size(0), outchannel, h, w))
#     output2 = torch.zeros((input.size(0), outchannel, h, w))
#     for i in range(0, h-overlap, size-overlap):
#         r_end = i + size if i + size < h else h
#         ind1_s = i + overlap // 2 if i > 0 else 0
#         ind1_e = i + size - overlap // 2 if i + size < h else h
#         for j in range(0, w-overlap, size-overlap):
#             c_end = j+size if j+size < w else w

#             input_patch = input[:,:,i:r_end,j:c_end]
#             input_var = input_patch.cuda()
#             with torch.no_grad():
#                 output_patch1, output_patch2 = model(input_var)
#                 #_, output_patch = model(input_var)

#             ind2_s = j+overlap//2 if j>0 else 0
#             ind2_e = j+size-overlap//2 if j+size<w else w
#             output1[:,:,ind1_s:ind1_e, ind2_s:ind2_e] = output_patch1[:,:,ind1_s-i:ind1_e-i, ind2_s-j:ind2_e-j]
#             output2[:,:,ind1_s:ind1_e, ind2_s:ind2_e] = output_patch2[:,:,ind1_s-i:ind1_e-i, ind2_s-j:ind2_e-j]

#     output1 = output1[:,:,:h0,:w0].cuda()
#     output2 = output2[:,:,:h0,:w0].cuda()

#     return output1, output2


# def post_proc(output, cutoff=0.5, cutoff_instance_max=0.3, cutoff_instance_avg=0.2, post_dilation_iter=2, post_fill_holes=True):
#     """
#     Split 1-channel merged output for instance segmentation
#     :param cutoff:
#     :param output: (h, w, 1) segmentation image
#     :return: list of (h, w, 1). instance-aware segmentations.
#     """
#     # The post processing function 'post_proc' is borrowed from the author of CIA-Net.
    
#     cutoffed = output > cutoff
#     lab_img = measure.label(cutoffed, connectivity=1)
#     instances = []
#     # pdb.set_trace()
#     for i in range(1, lab_img.max() + 1):
#         instances.append((lab_img == i).astype(bool))

#     filtered_instances = []
#     scores = []
#     for instance in instances:
#         # TODO : max or avg?
#         instance_score_max = np.max(instance * output)    # score max
#         if instance_score_max < cutoff_instance_max:
#             continue
#         instance_score_avg = np.sum(instance * output) / np.sum(instance)   # score avg
#         if instance_score_avg < cutoff_instance_avg:
#             continue
#         filtered_instances.append(instance)
#         scores.append(instance_score_avg)
#     instances = filtered_instances

#     # dilation
#     instances_tmp = []
#     if post_dilation_iter > 0:
#         for instance in filtered_instances:
#             instance = ndimage.morphology.binary_dilation(instance, iterations=post_dilation_iter)
#             instances_tmp.append(instance)
#         instances = instances_tmp

#     # sorted by size
#     sorted_idx = [i[0] for i in sorted(enumerate(instances), key=lambda x: get_size_of_mask(x[1]))]
#     instances = [instances[x] for x in sorted_idx]
#     scores = [scores[x] for x in sorted_idx]

#     # make sure there are no overlaps
#     # todo: this dataset gt has overlap, so do not use this func
#     # instances, scores = remove_overlaps(instances, scores)

#     # fill holes
#     if post_fill_holes:
#         instances = [ndimage.morphology.binary_fill_holes(i) for i in instances]
    
#     # instances = [np.expand_dims(i, axis=2) for i in instances]
#     # scores = np.array(scores)
#     # scores = np.expand_dims(scores, axis=1)
#     lab_img = np.zeros(instances[0].shape, dtype=np.int32)
#     for i, instance in enumerate(instances):
#         lab_img = np.maximum(lab_img, instance * (i + 1))
        
#     return lab_img


# def remove_overlaps(instances, scores):
#     if len(instances) == 0:
#         return [], []
#     lab_img = np.zeros(instances[0].shape, dtype=np.int32)
#     for i, instance in enumerate(instances):
#         lab_img = np.maximum(lab_img, instance * (i + 1))
#     instances = []
#     new_scores = []
#     for i in range(1, lab_img.max() + 1):
#         instance = (lab_img == i).astype(bool)
#         if np.max(instance) == 0:
#             continue
#         instances.append(instance)
#         new_scores.append(scores[i - 1])
#     return instances, new_scores


# def get_rect_of_mask(img):
#     rows = np.any(img, axis=1)
#     cols = np.any(img, axis=0)
#     rmin, rmax = np.where(rows)[0][[0, -1]]
#     cmin, cmax = np.where(cols)[0][[0, -1]]
#     return rmin, rmax, cmin, cmax


# def get_size_of_mask(img):
#     if np.max(img) == 0:
#         return 0
#     rmin, rmax, cmin, cmax = get_rect_of_mask(img)
#     return max([rmax - rmin, cmax - cmin])


# def data_split(data, fold=0):
#     validnum = int(data.shape[0] * 0.2)
#     valstart = fold * validnum
#     valend = (fold + 1) * validnum
#     val_data = data[valstart:valend]
#     return val_data


if __name__ == '__main__':
    main()
