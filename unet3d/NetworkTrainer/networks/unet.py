import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LUConv(nn.Module):
    def __init__(self, in_chan, out_chan, act, norm):
        super(LUConv, self).__init__()
        self.conv1 = nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1)

        if norm == 'bn':
            self.bn1 = nn.BatchNorm3d(num_features=out_chan, momentum=0.1,affine=True)
        elif norm == 'gn':
            self.bn1 = nn.GroupNorm(num_groups=8, num_channels=out_chan, eps=1e-05, affine=True)
        elif norm == 'in':
            self.bn1 = nn.InstanceNorm3d(num_features=out_chan, momentum=0.1, affine=True)
        else:
            raise ValueError('normalization type {} is not supported'.format(norm))

        if act == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act == 'prelu':
            self.activation = nn.PReLU(out_chan)
        elif act == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise ValueError('activation type {} is not supported'.format(act))

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        return out


def _make_nConv(in_channel, depth, act,norm, double_chnnel=False):
    if double_chnnel:
        layer1 = LUConv(in_channel, 32 * (2 ** (depth+1)),act,norm)
        layer2 = LUConv(32 * (2 ** (depth+1)), 32 * (2 ** (depth+1)),act,norm)
    else:
        layer1 = LUConv(in_channel, 32*(2**depth),act,norm)
        layer2 = LUConv(32*(2**depth), 32*(2**depth)*2,act,norm)

    return nn.Sequential(layer1,layer2)


class DownTransition(nn.Module):
    def __init__(self, in_channel,depth, act,norm):
        super(DownTransition, self).__init__()
        self.ops = _make_nConv(in_channel, depth,act,norm)

    def forward(self, x):
        return self.ops(x)
    # def forward(self, x):
    #     if self.current_depth == 3:
    #         out = self.ops(x)
    #         out_before_pool = out
    #     else:
    #         out_before_pool = self.ops(x)
    #         out = self.maxpool(out_before_pool)
    #     return out, out_before_pool

class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, depth,act,norm):
        super(UpTransition, self).__init__()
        self.depth = depth
        self.up_conv = nn.ConvTranspose3d(inChans, outChans, kernel_size=2, stride=2)
        self.ops = _make_nConv(inChans+ outChans//2,depth, act,norm, double_chnnel=True)

    def forward(self, x, skip_x):
        out_up_conv = self.up_conv(x)
        concat = torch.cat((out_up_conv,skip_x),1)
        return self.ops(concat)


class OutputTransition(nn.Module):
    def __init__(self, inChans, n_labels):

        super(OutputTransition, self).__init__()
        self.final_conv = nn.Conv3d(inChans, n_labels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.final_conv(x))
        return out

class UNet3D(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self,num_classes = 1, input_channels=1, act='relu',norm='in'):
        super(UNet3D, self).__init__()

        self.maxpool = nn.MaxPool3d(2)

        self.down_tr64 = DownTransition(input_channels,0,act,norm)
        self.down_tr128 = DownTransition(64,1,act,norm)
        self.down_tr256 = DownTransition(128,2,act,norm)
        self.down_tr512 = DownTransition(256,3,act,norm)

        self.up_tr256 = UpTransition(512, 512,2,act,norm)
        self.up_tr128 = UpTransition(256,256, 1,act,norm)
        self.up_tr64 = UpTransition(128,128,0,act,norm)
        self.final = nn.Conv3d(64, num_classes, kernel_size=1)

    def forward(self, x):
        skip_out64 = self.down_tr64(x)
        skip_out128 = self.down_tr128(self.maxpool(skip_out64))
        skip_out256 = self.down_tr256(self.maxpool(skip_out128))
        out512 = self.down_tr512(self.maxpool(skip_out256))

        out_up_256 = self.up_tr256(out512,skip_out256)
        out_up_128 = self.up_tr128(out_up_256,skip_out128)
        out_up_64 = self.up_tr64(out_up_128, skip_out64)
        out = self.final(out_up_64)
        # return torch.sigmoid(out)
        return out


class UNet3D_ds(UNet3D):
    def __init__(self,num_classes=1):
        super(UNet3D_ds, self).__init__()
        self.seg1 = nn.Conv3d(128, num_classes, kernel_size=1)
        self.seg2 = nn.Conv3d(256, num_classes, kernel_size=1)
        self.seg3 = nn.Conv3d(512, num_classes, kernel_size=1)

    def forward(self, x):
        skip_out64 = self.down_tr64(x)
        skip_out128 = self.down_tr128(self.maxpool(skip_out64))
        skip_out256 = self.down_tr256(self.maxpool(skip_out128))
        out512 = self.down_tr512(self.maxpool(skip_out256))

        out_up_256 = self.up_tr256(out512,skip_out256)
        out_up_128 = self.up_tr128(out_up_256,skip_out128)
        out_up_64 = self.up_tr64(out_up_128, skip_out64)
        out = self.final(out_up_64)
        out1 = self.seg1(out_up_128)
        out2 = self.seg2(out_up_256)
        out3 = self.seg3(out512)
        return [out, out1, out2, out3]