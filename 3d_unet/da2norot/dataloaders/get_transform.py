from dataloaders.data_kit import *
from torchvision import transforms

def get_transform(opt, phase='train'):
    # define data transforms for validation 
    transform = transforms.Compose([
        CenterCrop(opt.model['input_size']),
        ToTensor()
    ])

    # define data transforms for training
    if 'da1' in opt.task:
        transform = transforms.Compose([
            RandomCrop(opt.model['input_size']),
            RandomNoise(),
            GammaAdjust(),
            ToTensor()
        ])

    elif 'da2' in opt.task or 'da4' in opt.task:
        transform = transforms.Compose([
            RandomScale([0.85, 1.25]),
            RandomCrop(opt.model['input_size']),
            # RandomRotation(),
            RandomMirroring(),
            ToTensor()
        ])
    return transform

