import sys
sys.path.append('../../')
import time
from unittest import result
import torch
import torch.nn as nn
import torch.nn.functional as F
from NetworkTrainer.networks.densenet import densenet121, densenet161, densenet169, densenet201

class dilated_conv(nn.Module):
    """ same as original conv if dilation equals to 1 """
    def __init__(self, in_channel, out_channel, kernel_size=3, dropout_rate=0.0, activation=F.relu, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, padding=dilation, dilation=dilation)
        self.norm = nn.BatchNorm2d(out_channel)
        self.activation = activation
        if dropout_rate > 0:
            self.drop = nn.Dropout2d(p=dropout_rate)
        else:
            self.drop = lambda x: x  # no-op

    def forward(self, x):
        # CAB: conv -> activation -> batch normal
        x = self.norm(self.activation(self.conv(x)))
        x = self.drop(x)
        return x


class ConvDownBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dropout_rate=0.0, dilation=1):
        super().__init__()
        self.conv1 = dilated_conv(in_channel, out_channel, dropout_rate=dropout_rate, dilation=dilation)
        self.conv2 = dilated_conv(out_channel, out_channel, dropout_rate=dropout_rate, dilation=dilation)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.pool(x), x


class ConvUpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dropout_rate=0.0, dilation=1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channel, in_channel // 2, 2, stride=2)
        self.conv1 = dilated_conv(in_channel // 2 + out_channel, out_channel, dropout_rate=dropout_rate, dilation=dilation)
        self.conv2 = dilated_conv(out_channel, out_channel, dropout_rate=dropout_rate, dilation=dilation)

    def forward(self, x, x_skip):
        x = self.up(x)
        H_diff = x.shape[2] - x_skip.shape[2]
        W_diff = x.shape[3] - x_skip.shape[3]
        x_skip = F.pad(x_skip, (0, W_diff, 0, H_diff), mode='reflect')
        x = torch.cat([x, x_skip], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# Transfer Learning ResNet as Encoder part of UNet
class DenseUNet(nn.Module):
    def __init__(self, net='dense121', seg_classes=2):
        super().__init__()
        # load weight of pre-trained resnet
        if net == 'dense121':
            self.backbone = densenet121()
            l = [64, 256, 512, 1024, 1024]
        elif net == 'dense161':
            self.backbone = densenet161()
            l = [96, 384, 768, 2112, 2208]
        elif net == 'dense169':
            self.backbone = densenet169()
            l = [64, 256, 512, 1280, 1664]
        elif net == 'dense201':
            self.backbone = densenet201()
            l = [64, 256, 512, 1792, 1920]
        else:
            raise ValueError('Unknown network architecture: {}'.format(net))

        # up conv
        self.u5 = ConvUpBlock(l[4], l[3], dropout_rate=0.1)
        self.u6 = ConvUpBlock(l[3], l[2], dropout_rate=0.1)
        self.u7 = ConvUpBlock(l[2], l[1], dropout_rate=0.1)
        self.u8 = ConvUpBlock(l[1], l[0], dropout_rate=0.1)
        # final conv
        self.seg = nn.ConvTranspose2d(l[0], seg_classes, 2, stride=2)

    def forward(self, x):
        # refer https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py
        s1, s2, s3, s4, x = self.backbone(x)
        x1 = self.u5(x, s4)
        x1 = self.u6(x1, s3)
        x1 = self.u7(x1, s2)
        x1 = self.u8(x1, s1)
        out = self.seg(x1)
        return out


if __name__=='__main__':
    x = torch.randn((2, 3, 256, 256))
    net = DenseUNet(net='dense201')
    pred = net(x)
    print(pred.shape)