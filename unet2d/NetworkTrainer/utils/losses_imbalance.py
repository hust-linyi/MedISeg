"""
In ISIC dataset, the label shape is (b, x, y)
In Kitti dataset, the label shape is (b, 1, x, y, z)
"""
import torch
import torch.nn as nn
import numpy as np


class CELoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        self.weight = weight
        self.reduction = reduction

    def __call__(self, y_pred, y_true):
        y_true = y_true.long()
        if self.weight is not None:
            self.weight = self.weight.to(y_pred.device)
        if len(y_true.shape) == 4:
            y_true = y_true[:, 0, ...]
        loss = nn.CrossEntropyLoss(weight=self.weight, reduction=self.reduction)
        return loss(y_pred, y_true)


class WCELoss(nn.Module):
    """
    This loss function is a pixel-wise cross entropy loss which means it assigns different
    weight to different pixels. It allows us to pay more attention to hard pixels.
    Reference: https://arxiv.org/abs/1911.11445

    """
    def __init__(self):
        super().__init__()
    
    def forward(self,y_pred,y_true,weight=None):
        y_true = y_true.long()
        if weight is None:
            weight = 1
        if len(y_true.shape) == 4:
            y_true = y_true[:, 0, ...]
        loss = nn.CrossEntropyLoss(reduction='none')
        return (loss(y_pred,y_true)*weight).mean()
        

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-8):
        super(DiceLoss, self).__init__()
        
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        # first convert y_true to one-hot format
        axis = identify_axis(y_pred.shape)
        y_pred = nn.Softmax(dim=1)(y_pred)
        tp, fp, fn, _ = get_tp_fp_fn_tn(y_pred, y_true, axis)
        intersection = 2 * tp + self.smooth
        union = 2 * tp + fp + fn + self.smooth
        dice = 1 - (intersection / union)
        return dice.mean()


class IOULoss(nn.Module):
    def __init__(self,smooth):
        super(IOULoss,self).__init__()
        self.smooth = smooth

    def forward(self,y_pred,y_true,weight=None):
        axis = identify_axis(y_pred.shape)
        y_pred = nn.Softmax(dim=1)(y_pred)
        if weight is not None:
            weight = weight.to(y_pred.device)
        tp, fp, fn, _ = get_tp_fp_fn_tn(y_pred, y_true, axis,weight=weight)
        inter = tp
        union = tp + fp + fn
        iou = 1 - (inter + self.smooth) / (union + self.smooth)
        return iou.mean()


# taken from https://github.com/JunMa11/SegLoss/blob/master/test/nnUNetV2/loss_functions/focal_loss.py
class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=0.25, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, eps=1e-7):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, y_pred, y_true):
        axis = identify_axis(y_pred.shape)
        y_pred = nn.Softmax(dim=1)(y_pred)
        y_true = to_onehot(y_pred, y_true)
        y_pred = torch.clamp(y_pred, self.eps, 1. - self.eps)
        tp, fp, fn, _ = get_tp_fp_fn_tn(y_pred, y_true, axis)
        tversky = (tp + self.eps) / (tp + self.eps + self.alpha * fn + self.beta * fp)
        return (y_pred.shape[1] - tversky.sum()) / y_pred.shape[1]


class OHEMLoss(nn.CrossEntropyLoss):
    """
    Network has to have NO LINEARITY!
    """
    def __init__(self, weight=None, ignore_index=-100, k=0.7):
        super(OHEMLoss, self).__init__()
        self.k = k
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, y_pred, y_true):
        res = CELoss(reduction='none')(y_pred, y_true)
        num_voxels = np.prod(res.shape, dtype=np.int64)
        res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k), sorted=False)
        return res.mean()


def to_onehot(y_pred, y_true):
    shp_x = y_pred.shape
    shp_y = y_true.shape
    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            y_true = y_true.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(y_pred.shape, y_true.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = y_true 
        else:
            y_true = y_true.long()
            y_onehot = torch.zeros(shp_x, device=y_pred.device)
            y_onehot.scatter_(1, y_true, 1)
    return y_onehot



def get_tp_fp_fn_tn(net_output, gt, axes=None, square=False, weight=None):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    y_onehot = to_onehot(net_output, gt)

    if weight is None:
        weight = 1
    tp = net_output * y_onehot * weight
    fp = net_output * (1 - y_onehot) * weight
    fn = (1 - net_output) * y_onehot * weight
    tn = (1 - net_output) * (1 - y_onehot) * weight

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def identify_axis(shape):
    """
    Helper function to enable loss function to be flexibly used for 
    both 2D or 3D image segmentation - source: https://github.com/frankkramer-lab/MIScnn
    """
    # Three dimensional
    if len(shape) == 5 : return [2,3,4]
    # Two dimensional
    elif len(shape) == 4 : return [2,3]
    # Exception - Unknown
    else : raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')


if __name__=='__main__':
    y_pred = torch.rand(1, 3, 5, 5, 5)
    y_true = torch.randint(0, 3, (1, 5, 5, 5))
    # loss = DiceLoss()
    # loss = FocalLoss()
    loss = TverskyLoss()
    # loss = OHEMLoss()
    print(loss(y_pred, y_true))