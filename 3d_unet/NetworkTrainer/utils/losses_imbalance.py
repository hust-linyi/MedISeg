from secrets import token_hex
import torch
import torch.nn as nn
import numpy as np


class CELoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        self.weight = weight
        self.reduction = reduction

    def __call__(self, y_pred, y_true):
        y_true = y_true.long()
        self.weight = self.weight.to(y_pred.device)
        if len(y_true.shape) == 5:
            y_true = y_true[:, 0, ...]
        loss = nn.CrossEntropyLoss(weight=self.weight, reduction=self.reduction)
        return loss(y_pred, y_true)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-8):
        super(DiceLoss, self).__init__()
        
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        # first convert y_true to one-hot format
        axis = identify_axis(y_pred.shape)

        tp, fp, fn, _ = get_tp_fp_fn_tn(y_pred, y_true, axis)
        intersection = 2 * tp + self.smooth
        union = 2 * tp + fp + fn + self.smooth
        dice = 1 - (intersection / union)
        return dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.7, gamma=2., eps=1e-7):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
    def forward(self, y_pred, y_true):
        axis = identify_axis(y_pred.shape)
        y_true = to_onehot(y_pred, y_true)
        y_pred = torch.clamp(y_pred, self.eps, 1. - self.eps)
        cross_entropy = -y_true * torch.log(y_pred)
        loss = self.alpha * torch.pow(1 - y_pred, self.gamma) * cross_entropy
        return loss.mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=1e-8, eps=1e-7):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, y_pred, y_true):
        axis = identify_axis(y_pred.shape)
        y_true = to_onehot(y_pred, y_true)
        y_pred = torch.clamp(y_pred, self.beta, 1. - self.beta)
        tp, fp, fn, _ = get_tp_fp_fn_tn(y_pred, y_true, axis)
        tversky = (tp + self.eps) / (tp + self.eps + self.alpha * fn + self.beta * fp)
        return y_pred.shape[1] - tversky.sum()

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
        res = CELoss()(y_pred, y_true)
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



def get_tp_fp_fn_tn(net_output, gt, axes=None, square=False):
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

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

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
    # loss = TverskyLoss()
    loss = OHEMLoss()
    print(loss(y_pred, y_true))