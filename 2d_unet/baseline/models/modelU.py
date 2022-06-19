import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import resnet34, resnet101
from models.resnet1 import resnet34 as Resnet34


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
class ResUNet34(nn.Module):
    def __init__(self, net='res34', seg_classes = 2, colour_classes = 3, fixed_feature=False, pretrained=False):
        super().__init__()
        # load weight of pre-trained resnet
        self.resnet = resnet34(pretrained=pretrained)
        l = [64, 64, 128, 256, 512]
        if net == 'res101':
            self.resnet = resnet101(pretrained=pretrained)
            l = [64, 256, 512, 1024, 2048]
        # self.resnet1 = Resnet34(pretrained=False)
        if fixed_feature:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # up conv
        self.u5 = ConvUpBlock(l[4], l[3], dropout_rate=0.1)
        self.u6 = ConvUpBlock(l[3], l[2], dropout_rate=0.1)
        self.u7 = ConvUpBlock(l[2], l[1], dropout_rate=0.1)
        self.u8 = ConvUpBlock(l[1], l[0], dropout_rate=0.1)
        # final conv
        self.seg = nn.ConvTranspose2d(l[0], seg_classes, 2, stride=2)
        self.bnd = nn.ConvTranspose2d(l[0], seg_classes, 2, stride=2)

        self.colour = nn.ConvTranspose2d(l[0], colour_classes, 2, stride=2)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim = 1)
    def forward(self, x):
        # refer https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = s1 = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        # print(x.shape)
        x = s2 = self.resnet.layer1(x)
        # print(x.shape)
        x = s3 = self.resnet.layer2(x)
        # print(x.shape)
        x = s4 = self.resnet.layer3(x)
        # print(x.shape)
        x = self.resnet.layer4(x)
        # print(x.shape)

        x1 = self.u5(x, s4)
        x1 = self.u6(x1, s3)
        x1 = self.u7(x1, s2)
        x1 = self.u8(x1, s1)
        out = self.seg(x1)
        return out
