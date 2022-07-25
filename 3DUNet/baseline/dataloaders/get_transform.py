from dataloaders.data_kit import *
from torchvision import transforms

def get_transform(opt, phase='train'):
    # define data transforms for validation 
    if phase != 'train':
        transform = transforms.Compose([
            CenterCrop(opt.model['input_size']),
            ToTensor()
        ])
        
    else:
        # define data transforms for training
        if opt.task == 'da1':
            transform = transforms.Compose([
                RandomCrop(opt.model['input_size']),
                RandomNoise(),
                GammaAdjust(),
                ToTensor()
            ])

        elif opt.task == 'da2' or opt.task == 'da4':
            transform = transforms.Compose([
                RandomScale([0.85, 1.25]),
                RandomCrop(opt.model['input_size']),
                RandomRotation(),
                RandomMirroring(),
                ToTensor()
            ])
        else:
            transform = transforms.Compose([
                RandomCrop(opt.model['input_size']),
                ToTensor()
            ])
    return transform

