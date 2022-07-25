import sys
sys.path.append('../')
from NetworkTrainer.networks.unet import UNet
from NetworkTrainer.networks.resunet import ResUNet
from NetworkTrainer.networks.denseunet import DenseUNet
from NetworkTrainer.networks.resunet_ds import ResUNet_ds
from NetworkTrainer.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from NetworkTrainer.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import torch
from thop import profile
from rich import print


def getflops(model_name):
    if 'res' in model_name:
        net = ResUNet(net=model_name, seg_classes=2, colour_classes=3)
    elif 'dense' in model_name:
        net = DenseUNet(net=model_name, seg_classes=2)
    elif 'trans' in model_name:
        config_vit = CONFIGS_ViT_seg[model_name]
        config_vit.n_classes = 2
        config_vit.n_skip = 4
        if model_name.find('R50') != -1:
            config_vit.patches.grid = (int(256 / 16), int(256 / 16))
        net = ViT_seg(config_vit, img_size=256, num_classes=config_vit.n_classes).cuda()
    else:
        net = UNet(3, 2, 2)
    input1 = torch.randn(4, 3, 224, 224) 
    flops, params = profile(net, inputs=(input1, ))
    return flops, params
    


def get_flops():
    model_list = ['res18', 'res34', 'res50', 'res101', 'res152',
    'dense121', 'dense161', 'dense169', 'dense201',
    'ViT-B_16', 'ViT-B_32', 'ViT-L_16', 'ViT-L_32', 'ViT-H_14', 'R50-ViT-B_16', 'R50-ViT-L_16']
    flops_list, params_list = [], []
    for model_name in model_list:
        flops, params = getflops(model_name)
        flops_list.append(flops)
        params_list.append(params)
    for i in range(len(model_list)):
        print(f'{model_list[i]}:\t{flops_list[i]/1e9:.2f} GFLOPs, {params_list[i]/1e6:.2f} Mparams')

get_flops()