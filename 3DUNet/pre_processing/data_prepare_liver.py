
from collections import OrderedDict
import os
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
import SimpleITK as sitk
from collections import OrderedDict
from batchgenerators.augmentations.spatial_transformations import augment_resize
from rich import print
import tqdm


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
    def __init__(self, downloaded_data_dir= "",task_name = "", out_data_dir="" ):
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


    def get_raw_training_data(self):
        imagestr = join(self.out_base_raw, "imagesTr")
        maybe_mkdir_p(imagestr)
        patient_ids = [f.replace('segmentation-', '').replace('.nii', '') for f in os.listdir(self.downloaded_data_dir) if 'segmentation' in f]
        print(f'Got total {len(patient_ids)} raw data')
        patient_ids.sort()
        for patient_id in tqdm.tqdm(patient_ids):
            img_1 = sitk.ReadImage(join(self.downloaded_data_dir, 'volume-' + patient_id + '.nii'))
            img_2 = sitk.ReadImage(join(self.downloaded_data_dir, 'segmentation-' + patient_id + '.nii'))

            # set direction
            img_1.SetDirection(tuple((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)))
            img_2.SetDirection(tuple((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)))

            volume = sitk.GetArrayFromImage(img_1) # z, x, y
            label = sitk.GetArrayFromImage(img_2) # z, x, y
            np.save(join(imagestr, patient_id + "_image.npy"),volume.astype(np.float32))
            np.save(join(imagestr, patient_id + "_label.npy"),label)

            ori1, spacing1, direction1, size1 = img_1.GetOrigin(), img_1.GetSpacing(), img_1.GetDirection(), img_1.GetSize()

            if True:
                self.train_patient_names.append(patient_id)
                # self.images.append(volume)
                # self.labels.append(label)
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


    def resample(self, image, label, spacing, new_spacing=[1,1,1]):
        spacing, new_spacing = np.array(spacing), np.array(new_spacing)
        resize_factor = spacing / new_spacing
        old_shape = np.array(image.shape)
        new_real_shape = old_shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / old_shape
        new_spacing = spacing / real_resize_factor

        # image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
        image = np.moveaxis(image, 0, -1) # x, y, z
        label = np.moveaxis(label, 0, -1) # x, y, z
        image = np.expand_dims(image, axis=0) # 1, x, y, z
        label = np.expand_dims(label, axis=0) # 1, x, y, z

        # target_size = tuple(new_shape.transpose(1,2,0).astype(int)) # x, y, z
        target_size = tuple(np.array((new_shape[1], new_shape[2], new_shape[0])).astype(int)) # x, y, z
        out_img, out_seg = augment_resize(sample_data=image, sample_seg=label, target_size=target_size)
        out_img, out_seg = out_img[0], out_seg[0] # x,y,z
        return out_img, out_seg


    def do_preprocessing(self,minimun=0, maxmun=0, new_spacing=(3.22,1.62,1.62)):
        maybe_mkdir_p(self.out_base_preprocess)
        self.data_info = pickle.load(open(join(self.out_base_raw, 'dataset_pro.pkl'), 'rb'))
        for i in range(len(self.data_info['patient_names'])):
            print(f"Preprocessing {i}/{len(self.data_info['patient_names'])}")
            # voxels = self.images[i]
            # label = self.labels[i]
            voxels = np.load(join(self.out_base_raw, "imagesTr", self.data_info['patient_names'][i] + "_image.npy"))
            label = np.load(join(self.out_base_raw, "imagesTr", self.data_info['patient_names'][i] + "_label.npy"))
            if minimun:
                lower_bound = minimun
                upper_bound = maxmun 
            else:
                upper_bound,lower_bound,median,mean_before,sd_before = self._get_voxels_in_foreground(voxels,label)
            mask = (voxels > lower_bound) & (voxels < upper_bound)
            voxels = np.clip(voxels, lower_bound, upper_bound)
            # mn = voxels[mask].mean()
            # sd = voxels[mask].std()
            # voxels = (voxels - mn) / sd

            ### Convert to [0, 1]
            voxels = (voxels - voxels.min()) / (voxels.max() - voxels.min())
            
            # resample to isotropic voxel size
            spacing = self.data_info['dataset_properties'][self.data_info['patient_names'][i]]['spacing']
            spacing = (spacing[2], spacing[0], spacing[1])

            voxels, label = self.resample(voxels, label, spacing, new_spacing)
            np.save(join(self.out_base_preprocess, self.data_info['patient_names'][i] + "_image.npy"),voxels.astype(np.float32))
            np.save(join(self.out_base_preprocess, self.data_info['patient_names'][i] + "_label.npy"),label)
        save_pickle(self.data_info, join(self.out_base_preprocess, 'dataset_pro.pkl'))
        with open(self.out_base_preprocess + '/all.txt', 'w') as f :  
            for train_patient in self.data_info['patient_names']:
                f.write(train_patient)
                f.write('\n')

if __name__ == "__main__":
    # SM_server
    downloaded_data_dir = "/mnt/yfs/ianlin/Data/LIVER/RAWDATA"
    out_data_dir = "/mnt/yfs/ianlin/Data/LIVER/preprocess"

    # eez244
    # downloaded_data_dir = "/home/ylindq/Data/LIVER/RAWDATA"
    # out_data_dir = "/home/ylindq/Data/LIVER/"

    cada = GenericPreprocessor(downloaded_data_dir=downloaded_data_dir, out_data_dir=out_data_dir,task_name="monai")
    # cada.get_raw_training_data()
    cada.do_preprocessing(minimun=-124, maxmun=276, new_spacing=(1, 1, 1))


