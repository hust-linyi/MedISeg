from NetworkTrainer.dataloaders.dataload import *

def get_transform(opt, phase='train'):
    # define data transforms for validation 
    if phase != 'train':
        transform = [
            CenterCrop(opt.model['input_size']),
            ToTensor()
        ]
        
    else:
        # define data transforms for training
        if opt.task == 'da1':
            transform = [
                RandomCrop(opt.model['input_size']),
                RandomNoise(),
                GammaAdjust(),
                ToTensor()
            ]

        elif opt.task == 'da2':
            transform = [
                RandomScale([0.85, 1.25]),
                RandomCrop(opt.model['input_size']),
                RandomRotation(),
                RandomMirroring(),
                ToTensor()
            ]
        elif opt.task == 'da2flip':
            transform = [
                RandomCrop(opt.model['input_size']),
                RandomMirroring(),
                ToTensor()
            ]
        elif opt.task == 'da2rot':
            transform = [
                RandomCrop(opt.model['input_size']),
                RandomRotation(),
                ToTensor()
            ]
        elif opt.task == 'da2scale':
            transform = [
                RandomScale([0.85, 1.25]),
                RandomCrop(opt.model['input_size']),
                ToTensor()
            ]
        elif opt.task == 'da4':
            transform = [
                RandomScale([0.85, 1.25]),
                RandomCrop(opt.model['input_size']),
                RandomRotation(),
                RandomMirroring(),
                RandomNoise(),
                GammaAdjust(),
                ToTensor()
            ]
        elif opt.task == 'oversample':
            transform = [
                SelectedCrop(opt.model['input_size']),
                ToTensor()
            ]
        else:
            transform = [
                RandomCrop(opt.model['input_size']),
                ToTensor()
            ]
    return transform

