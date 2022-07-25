import torch
import torch.nn.functional as F
import math

def dice_loss(input, target, eps=1e-7, if_sigmoid=True):
    if if_sigmoid:
        input = F.sigmoid(input)
    b = input.shape[0]
    iflat = input.contiguous().view(b, -1)
    tflat = target.float().contiguous().view(b, -1)
    intersection = (iflat * tflat).sum(dim=1)
    L = (1 - ((2. * intersection + eps) / (iflat.pow(2).sum(dim=1) + tflat.pow(2).sum(dim=1) + eps))).mean()
    return L

def smooth_truncated_loss(p, t, ths=0.06, if_reduction=True, if_balance=True):
    n_log_pt = F.binary_cross_entropy_with_logits(p, t, reduction='none')
    pt = (-n_log_pt).exp()
    L = torch.where(pt>=ths, n_log_pt, -math.log(ths)+0.5*(1-pt.pow(2)/(ths**2)))
    if if_reduction:
        if if_balance:
            return 0.5*((L*t).sum()/t.sum().clamp(1) + (L*(1-t)).sum()/(1-t).sum().clamp(1))
        else:
            return L.mean()
    else:
        return L

def balance_bce_loss(input, target):
    L0 = F.binary_cross_entropy_with_logits(input, target, reduction='none')
    return 0.5*((L0*target).sum()/target.sum().clamp(1)+(L0*(1-target)).sum()/(1-target).sum().clamp(1))

def compute_loss_list(loss_func, pred=[], target=[], **kwargs):
    losses = []
    for ipred, itarget in zip(pred, target):
        losses.append(loss_func(ipred, itarget, **kwargs))
    return losses