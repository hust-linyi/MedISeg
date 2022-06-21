import torch
from NetworkTrainer.networks.unet import UNet3D
from NetworkTrainer.utils.test_util import test_all_case
from NetworkTrainer.dataloaders.data_kit import get_imglist
import os

def test_calculate_metric(opt):
    net = UNet3D(num_classes=3, input_channels=1, act='relu', norm=opt.train['norm'])
    net = torch.nn.DataParallel(net)
    net = net.cuda()

    print(f"=> loading trained model in {opt.test['model_path']}")
    checkpoint = torch.load(opt.test['model_path'])
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    image_list = get_imglist(opt.root_dir, opt.fold, 'test')
    image_list = [os.path.join(opt.root_dir, img) for img in image_list]

    test_all_case(net, image_list, num_classes=3,
                               patch_size=opt.model['input_size'], stride_xy=opt.model['input_size'][0]//2, stride_z=opt.model['input_size'][0]//2,
                               save_result=opt.test['save_flag'], test_save_path=opt.test['save_dir'])
