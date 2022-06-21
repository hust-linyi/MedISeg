import torch
import os
from networks.unet import UNet3D
from utils.test_util import test_all_case
from dataloaders.data_kit import get_imglist
from options import Options


def test_calculate_metric(opt):
    net = UNet3D(num_classes=3, input_channels=1, act='relu', norm=opt.train['norm'])
    net = torch.nn.DataParallel(net)
    net = net.cuda()

    print(f"=> loading trained model in {opt.test['model_path']}")
    checkpoint = torch.load(opt.test['model_path'])
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    image_list = get_imglist(opt.root_dir, opt.fold)
    test_all_case(net, image_list, num_classes=3,
                               patch_size=opt.model['input_size'], stride_xy=opt.model['input_size'][0]//2, stride_z=opt.model['input_size'][0]//2,
                               save_result=True, test_save_path=opt.test['save_dir'])


if __name__ == '__main__':
    opt = Options(isTrain=False)
    opt.parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.test['gpus'])
    metric = test_calculate_metric(opt)
