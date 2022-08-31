import pandas as pd
import os


def get_imglist(root_dir, fold, phase):
    image_list = [item.replace('_image.npy','') for item in os.listdir(root_dir) if item.endswith('_image.npy')]
    testnum = int(len(image_list) * 0.2)
    teststart = fold * testnum
    testend = (fold + 1) * testnum
    if phase == 'test':
        image_list = image_list[teststart:testend]
    else:
        image_list = np.concatenate([image_list[:teststart], image_list[testend:]], axis=0)
        valnum = int(len(image_list) * 0.1)
    if phase == 'train':
        image_list = image_list[valnum:]
    elif phase == 'val':
        valnum = int(len(image_list) * 0.1)
        image_list = image_list[:valnum]
    return image_list


folds = [0, 1, 2, 3, 4]
phase = ['train', 'val', 'test']
root_dir = '/newdata/ianlin/Data/COVID-19-20/monai/preprocess'
out_dir = '/newdata/ianlin/Experiment/COVID/tmp'
for f in folds:
    for p in phase:
        image_list = get_imglist(root_dir, f, p)
        image_list_pd = pd.DataFrame(image_list)
        image_list_pd.to_csv(os.path.join(out_dir, f'fold{f}_{p}.csv'), index=False, header=False)
