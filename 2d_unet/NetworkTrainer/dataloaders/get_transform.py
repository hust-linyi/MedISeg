import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transform(opt, phase='train'):
    # define data transforms for validation 
    if phase != 'train':
        transform = [
            A.Resize(opt.model['input_size'][1], opt.model['input_size'][0]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
        
    else:
        # define data transforms for training
        if opt.task == 'da1':
            transform = [
                A.Resize(opt.model['input_size'][1], opt.model['input_size'][0]),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                A.CLAHE(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]

        elif opt.task == 'da2':
            transform = [
                A.Resize(opt.model['input_size'][1], opt.model['input_size'][0]),
                A.PadIfNeeded(min_height=opt.model['input_size'][0], min_width=opt.model['input_size'][1]),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
                A.RandomCrop(height=opt.model['input_size'][0], width=opt.model['input_size'][1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        elif opt.task == 'da4':
            transform = [
                A.Resize(opt.model['input_size'][1], opt.model['input_size'][0]),
                A.PadIfNeeded(min_height=opt.model['input_size'][0], min_width=opt.model['input_size'][1]),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
                A.RandomCrop(height=opt.model['input_size'][0], width=opt.model['input_size'][1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                A.CLAHE(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        else:
            transform = [
                A.Resize(opt.model['input_size'][1], opt.model['input_size'][0]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
    return transform

