import torch
#torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.utils as utils
import torch.nn.init as init
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from models.densenet import densenet121_backbone
import re

class Conv_Block(nn.Module):
    def __init__(self, din, dout, act_fn, kernel_size=3, stride=1, padding=1, dilation=1):
        super(Conv_Block,self).__init__()
        self.bn = nn.BatchNorm2d(din)
        self.act_fn = act_fn
        self.conv = nn.Conv2d(din, dout, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    def forward(self,input):
        return self.conv(self.act_fn(self.bn(input)))

class UNet(nn.Module):
    def __init__(self, input_nc, nd_sout, nd_cout, nd_decoder=256, if_from_scratch=False):
        super(UNet,self).__init__()

        act_fn = nn.ReLU(inplace=True)
       
        self.conv_fdim_trans_3_s = Conv_Block(1024, nd_decoder, act_fn, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv_fdim_trans_2_s = Conv_Block(1024, nd_decoder, act_fn, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv_fdim_trans_1_s = Conv_Block(512, nd_decoder, act_fn, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv_fdim_trans_0_s = Conv_Block(256, nd_decoder, act_fn, kernel_size=1, stride=1, padding=0, dilation=1)

        self.conv_fdim_trans_3_c = Conv_Block(1024, nd_decoder, act_fn, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv_fdim_trans_2_c = Conv_Block(1024, nd_decoder, act_fn, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv_fdim_trans_1_c = Conv_Block(512, nd_decoder, act_fn, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv_fdim_trans_0_c = Conv_Block(256, nd_decoder, act_fn, kernel_size=1, stride=1, padding=0, dilation=1)
        
        self.conv_0a_s = nn.Conv2d(nd_decoder, nd_decoder, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv_1a_s = nn.Conv2d(nd_decoder, nd_decoder, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv_2a_s = nn.Conv2d(nd_decoder, nd_decoder, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv_3a_s = nn.Conv2d(nd_decoder, nd_decoder, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv_0a_c = nn.Conv2d(nd_decoder, nd_decoder, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv_1a_c = nn.Conv2d(nd_decoder, nd_decoder, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv_2a_c = nn.Conv2d(nd_decoder, nd_decoder, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv_3a_c = nn.Conv2d(nd_decoder, nd_decoder, kernel_size=3, stride=1, padding=1, dilation=1)
        
        self.conv_0b = Conv_Block(nd_decoder*2, nd_decoder, act_fn, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv_1b = Conv_Block(nd_decoder*2, nd_decoder, act_fn, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv_2b = Conv_Block(nd_decoder*2, nd_decoder, act_fn, kernel_size=3, stride=1, padding=1, dilation=1)
        
        self.conv_0c_s = Conv_Block(nd_decoder, nd_decoder, act_fn, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv_1c_s = Conv_Block(nd_decoder, nd_decoder, act_fn, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv_2c_s = Conv_Block(nd_decoder, nd_decoder, act_fn, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv_0c_c = Conv_Block(nd_decoder, nd_decoder, act_fn, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv_1c_c = Conv_Block(nd_decoder, nd_decoder, act_fn, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv_2c_c = Conv_Block(nd_decoder, nd_decoder, act_fn, kernel_size=3, stride=1, padding=1, dilation=1)
        
        for m in self.modules():        
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        
        self.out_s_0 = nn.Conv2d(nd_decoder, nd_sout, kernel_size=1, stride=1, padding=0)
        self.out_c_0 = nn.Conv2d(nd_decoder, nd_cout, kernel_size=1, stride=1, padding=0)
        self.out_s_1 = nn.Conv2d(nd_decoder, nd_sout, kernel_size=1, stride=1, padding=0)
        self.out_c_1 = nn.Conv2d(nd_decoder, nd_cout, kernel_size=1, stride=1, padding=0)
        self.out_s_2 = nn.Conv2d(nd_decoder, nd_sout, kernel_size=1, stride=1, padding=0)
        self.out_c_2 = nn.Conv2d(nd_decoder, nd_cout, kernel_size=1, stride=1, padding=0)
        self.out_s_3 = nn.Conv2d(nd_decoder, nd_sout, kernel_size=1, stride=1, padding=0)
        self.out_c_3 = nn.Conv2d(nd_decoder, nd_cout, kernel_size=1, stride=1, padding=0)
        
        self.out_s = nn.Conv2d(nd_decoder, nd_sout, kernel_size=1, stride=1, padding=0)
        self.out_c = nn.Conv2d(nd_decoder, nd_cout, kernel_size=1, stride=1, padding=0)

        self.encoder_backbone = densenet121_backbone(True)
        
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

        if not(if_from_scratch):
            state_dict_pretrained = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/densenet121-a639ec97.pth')
            model_dict = self.encoder_backbone.state_dict()
            new_dict = {}
            pattern = re.compile(
                r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
            for key in list(state_dict_pretrained.keys()):
                res = pattern.match(key)
                if res:
                    new_key = res.group(1) + res.group(2)
                    state_dict_pretrained[new_key] = state_dict_pretrained[key]
                    del state_dict_pretrained[key]

            for k, v in state_dict_pretrained.items():
                if k in model_dict:
                    new_dict.update({k:v})
            model_dict.update(new_dict)
            self.encoder_backbone.load_state_dict(model_dict)
        else:
            for m in self.encoder_backbone.modules():        
                if isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, img):

        # nd_decoder, 512, 1024, 1024
        f0, f1, f2, f3 = self.encoder_backbone(img)
        
        f3s = self.conv_fdim_trans_3_s(f3)
        f3c = self.conv_fdim_trans_3_c(f3)
        f2s = self.conv_fdim_trans_2_s(f2)
        f2c = self.conv_fdim_trans_2_c(f2)
        f1s = self.conv_fdim_trans_1_s(f1)
        f1c = self.conv_fdim_trans_1_c(f1)
        f0s = self.conv_fdim_trans_0_s(f0)
        f0c = self.conv_fdim_trans_0_c(f0)
        
        conv_0a_s = self.conv_0a_s(f0s)
        conv_0a_c = self.conv_0a_c(f0c)
        conv_1a_s = self.conv_1a_s(f1s + self.avgpool(conv_0a_s))
        conv_1a_c = self.conv_1a_c(f1c + self.avgpool(conv_0a_c))
        conv_2a_s = self.conv_2a_s(f2s + self.avgpool(conv_1a_s))
        conv_2a_c = self.conv_2a_c(f2c + self.avgpool(conv_1a_c))
        conv_3a_s = self.conv_3a_s(f3s + self.avgpool(conv_2a_s))
        conv_3a_c = self.conv_3a_c(f3c + self.avgpool(conv_2a_c))
        
        conv_0b = self.conv_0b(torch.cat((conv_0a_s, conv_0a_c), dim=1))
        conv_1b = self.conv_1b(torch.cat((conv_1a_s, conv_1a_c), dim=1))
        conv_2b = self.conv_2b(torch.cat((conv_2a_s, conv_2a_c), dim=1))
        
        conv_2c_s = self.conv_2c_s(F.interpolate(conv_3a_s, scale_factor=2, mode='bilinear', align_corners=True)+conv_2b+conv_2a_s)
        conv_1c_s = self.conv_1c_s(F.interpolate(conv_2c_s, scale_factor=2, mode='bilinear', align_corners=True)+conv_1b+conv_1a_s)
        conv_0c_s = self.conv_0c_s(F.interpolate(conv_1c_s, scale_factor=2, mode='bilinear', align_corners=True)+conv_0b+conv_0a_s)

        conv_2c_c = self.conv_2c_c(F.interpolate(conv_3a_c, scale_factor=2, mode='bilinear', align_corners=True)+conv_2b+conv_2a_c)
        conv_1c_c = self.conv_1c_c(F.interpolate(conv_2c_c, scale_factor=2, mode='bilinear', align_corners=True)+conv_1b+conv_1a_c)
        conv_0c_c = self.conv_0c_c(F.interpolate(conv_1c_c, scale_factor=2, mode='bilinear', align_corners=True)+conv_0b+conv_0a_c)
        
        sout = self.out_s(conv_0c_s)
        cout = self.out_c(conv_0c_c)
        
        sout_0 = self.out_s_0(conv_0a_s)
        cout_0 = self.out_c_0(conv_0a_c)
        sout_1 = self.out_s_1(conv_1a_s)
        cout_1 = self.out_c_1(conv_1a_c)
        sout_2 = self.out_s_2(conv_2a_s)
        cout_2 = self.out_c_2(conv_2a_c)
        sout_3 = self.out_s_3(conv_3a_s)
        cout_3 = self.out_c_3(conv_3a_c)

        # return sout, sout_0, sout_1, sout_2, sout_3, cout, cout_0, cout_1, cout_2, cout_3
        return sout, cout
        
    # def get_last_shared_layer(self):
    #     return f1, f2, f3, f4
    