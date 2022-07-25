
from collections import OrderedDict
import sys
import os
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from multiprocessing import Pool
import nibabel
import numpy as np
import SimpleITK as sitk
from collections import OrderedDict
import scipy
import tarfile


def load_itk_image(path):
    # import ipdb;ipdb.set_trace()
    for root, dirs, _ in os.walk(path):
        dirs.sort()
        if dirs:
            path = os.path.join(root, dirs[0])
        else:
            break
    series_IDs = sitk.ImageSeriesReader_GetGDCMSeriesIDs(path)
    series_filenames = sitk.ImageSeriesReader_GetGDCMSeriesFileNames(path, series_IDs[0])
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_filenames)
    itkimage = series_reader.Execute()
    return itkimage


# sitk.ReadImage(expected_label_file)
class GenericPreprocessor(object):
    def __init__(self, downloaded_data_dir= "/apdcephfs/share_1290796/yunqiaoyang/CADA",task_name = "CADA_seg", \
          out_data_dir = r"E:\project\DATA\CT\data_10.19\preprocess" ):
        self.downloaded_data_dir = downloaded_data_dir
        self.out_data_dir = out_data_dir
        self.task_name = task_name
        self.data_info = OrderedDict()
        self.data_info['patient_names'] = []
        self.data_info['dataset_properties'] = OrderedDict()  
        self.train_patient_names = []

        self.out_base_raw = os.path.join(self.out_data_dir,self.task_name,'raw')
        self.out_base_preprocess = os.path.join(self.out_data_dir,self.task_name,'preprocess')
        if not os.path.exists(self.out_base_preprocess):
            os.makedirs(self.out_base_preprocess)
        if not os.path.exists(self.out_base_raw):
            os.makedirs(self.out_base_raw)

        self.images = []
        self.labels = []


    def search_label(self, label_dir):
        # import ipdb;ipdb.set_trace()

        content = os.listdir(label_dir)
        filename = '.tar'
        out = None
        for each in content:
            each_path = label_dir + os.sep + each
            if filename in each:
                return each_path
            if os.path.isdir(each_path):
                out = self.search_label(each_path)
        return out

    def get_raw_training_data(self):

        imagestr = join(self.out_base_raw, "imagesTr")
        maybe_mkdir_p(imagestr)

        folder_data = join(self.downloaded_data_dir, "data")
        folder_labels = join(self.downloaded_data_dir, "label")

        # for p in subfiles(folder_data, join=False):
        for p in os.listdir(folder_data):
            patient_id = p
            # import ipdb;ipdb.set_trace()

            label_dir = self.search_label(join(folder_data, patient_id))
            if not label_dir:
                print(f'{patient_id} miss label!')
                continue

            tar = tarfile.open(label_dir, 'r:tar')
            file_names = tar.getnames()
            if len(file_names) != 2:
                print(f'{patient_id} miss label!')
                continue
            for file_name in file_names:
                tar.extract(file_name, folder_labels)
            tar.close()
            os.rename(os.path.join(folder_labels, 'IMG00000_DCM_Label.nii.gz'),
                      os.path.join(folder_labels, patient_id + '.nii.gz'))
            os.rename(os.path.join(folder_labels, 'IMG00000_DCM_Label.json'),
                      os.path.join(folder_labels, patient_id + '.json'))

            # volume = nibabel.load(join(folder_data, patient_id +"_orig.nii.gz")).get_data()
            # img_1 = sitk.ReadImage(join(folder_data, patient_id +"_orig.nii.gz"))
            img_1 = load_itk_image(join(folder_data, patient_id))
            # img1 = load_itk_image()
            volume = sitk.GetArrayFromImage(img_1)
            print(volume.shape)
            np.save(join(imagestr, patient_id + "_image.npy"),volume.astype(np.float32))

            # label = nibabel.load(join(folder_labels, patient_id +"_masks.nii.gz")).get_data()


            try:
                img_2 = sitk.ReadImage(join(folder_labels, patient_id +".nii.gz"))
            except:
                continue
            # img_2 = load_itk_image(join(folder_labels, patient_id))
            label = sitk.GetArrayFromImage(img_2)

            label[label == 0] = 1
            label = label - 1
            print("min = {},max = {}".format(np.min(label), np.max(label)))

            np.save(join(imagestr, patient_id + "_label.npy"),label)

            ori1, spacing1, direction1, size1 = img_1.GetOrigin(), img_1.GetSpacing(), img_1.GetDirection(), img_1.GetSize()

            if True:
                self.train_patient_names.append(patient_id)
                self.images.append(volume)
                self.labels.append(label)
                self.data_info['dataset_properties'][patient_id] = OrderedDict()  
                self.data_info['dataset_properties'][patient_id]['origin'] = ori1
                self.data_info['dataset_properties'][patient_id]['spacing'] = spacing1
                self.data_info['dataset_properties'][patient_id]['direction'] = direction1
                self.data_info['dataset_properties'][patient_id]['size'] = size1
            
        self.data_info['patient_names'] = self.train_patient_names

        # with open(join(out_base_raw, 'dataset_pro.pkl'), 'wb') as f:
        #     pickle.dump(self.data_info, f)
        save_pickle(self.data_info, join(self.out_base_raw, 'dataset_pro.pkl'))

        # train_patient_names.sort()
        with open(imagestr + '/train.txt', 'w') as f :  
            for train_patient in self.train_patient_names:
                f.write(train_patient)
                f.write('\n')


    def _get_voxels_in_foreground(self,voxels,label):
        mask = label> 0
        # image = list(voxels[mask][::10]) # no need to take every voxel
        image = list(voxels[mask])
        median = np.median(image)
        mean = np.mean(image)
        sd = np.std(image)
        percentile_99_5 = np.percentile(image, 99.5)
        percentile_00_5 = np.percentile(image, 00.5)
        return percentile_99_5,percentile_00_5, median,mean,sd


    def resample(self, image, spacing, new_spacing=[1,1,1]):
        resize_factor = spacing / new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        new_spacing = spacing / real_resize_factor

        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

        return image, new_spacing

    def do_preprocessing(self,minimun=0, maxmun=0):
        maybe_mkdir_p(self.out_base_preprocess)
        # import ipdb;ipdb.set_trace()
        for i in  range(len(self.train_patient_names)):
            voxels = self.images[i]
            label = self.labels[i]
            # spacing = self.data_info['dataset_properties'][self.train_patient_names[i]]['spacing']
            # voxels, new_spacing = self.resample(voxels, spacing)
            # label, _ = self.resample(label, spacing)
            # self.data_info['dataset_properties'][self.train_patient_names[i]]['spacing'] = new_spacing
            if minimun:
                lower_bound = minimun
                upper_bound = maxmun 
            else:
                upper_bound,lower_bound,median,mean_before,sd_before = self._get_voxels_in_foreground(voxels,label)
            mask = (voxels > lower_bound) & (voxels < upper_bound)
            voxels = np.clip(voxels, lower_bound, upper_bound)
            mn = voxels[mask].mean()
            sd = voxels[mask].std()
            voxels = (voxels - mn) / sd
            print(self.train_patient_names[i])
            print("lower_bound = {},upper_bound = {}, before: mean = {}, std = {} after: mean = {}, std = {}".format(lower_bound, upper_bound,median, mean_before,sd_before, mn, sd))
            np.save(join(self.out_base_preprocess, self.train_patient_names[i] + "_image.npy"),voxels.astype(np.float32))
            np.save(join(self.out_base_preprocess, self.train_patient_names[i] + "_label.npy"),label)
        save_pickle(self.data_info, join(self.out_base_preprocess, 'dataset_pro.pkl'))
        with open(self.out_base_preprocess + '/all.txt', 'w') as f :  
            for train_patient in self.train_patient_names:
                f.write(train_patient)
                f.write('\n')

    # def do_split(self):
    #     maybe_mkdir_p(self.out_base_preprocess )
    #     splits_file = join(self.out_base_preprocess, "splits_final.pkl")
    #     if not isfile(splits_file):
    #         self.print_to_log_file("Creating new split...")
    #         splits = []
    #         all_keys_sorted = np.sort(list(self.dataset.keys()))
    #         kfold = KFold(n_splits=3) #, shuffle=True, random_state=12345)
    #         ####kold cross validation
    #         for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
    #             train_keys = np.array(all_keys_sorted)[train_idx]
    #             test_keys = np.array(all_keys_sorted)[test_idx]
    #             splits.append(OrderedDict())
    #             splits[-1]['train'] = train_keys
    #             splits[-1]['val'] = test_keys
    #         save_pickle(splits, splits_file)

    #     splits = load_pickle(splits_file)

    #     if self.fold == "all":
    #         tr_keys = val_keys = list(self.dataset.keys())
    #     else:
    #         tr_keys = splits[self.fold]['train']
    #         val_keys = splits[self.fold]['val']

    #     tr_keys.sort()
    #     val_keys.sort()

    #     self.dataset_tr = OrderedDict()
    #     for i in tr_keys:
    #         self.dataset_tr[i] = self.dataset[i]

    #     self.dataset_val = OrderedDict()
    #     for i in val_keys:
    #         self.dataset_val[i] = self.dataset[i]

if __name__ == "__main__":
    # cada = GenericPreprocessor(downloaded_data_dir= "E:\project\DATA\CT\data_10.20",
    #                            out_data_dir=r'E:\project\DATA\CT\data_10.20\preprocess',
    #                             task_name = "data_clean")
    cada = GenericPreprocessor(downloaded_data_dir= "/extracephonline/medai_data_hongyuzhou/ianylin/data/TB_CT/data_10.19",
                               out_data_dir=r'/extracephonline/medai_data_hongyuzhou/ianylin/data/TB_CT/data_10.19/preprocess',
                               task_name="data_clean2")
    cada.get_raw_training_data()
    cada.do_preprocessing()


