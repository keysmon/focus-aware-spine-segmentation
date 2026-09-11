"""Class-balanced hard-pixel BCE added to the existing focus objective."""
import math
import torch
from torch.nn import functional as F
from experiments.tversky import loss_value as primary_loss


def hard_pixel_loss(logits,target):
    if logits.shape!=target.shape or logits.ndim!=4 or not logits.numel():
        raise ValueError('Expected matching nonempty NCHW logits and binary targets')
    errors=F.binary_cross_entropy_with_logits(logits,target,reduction='none')
    means=[]
    for label in (0,1):
        values=errors[target==label]
        if values.numel():
            k=max(1,math.ceil(values.numel()*.25))
            means.append(torch.topk(values,k,sorted=False).values.mean())
    if not means:raise ValueError('Expected binary targets')
    return torch.stack(means).mean()


def loss_value(logits,target):
    return primary_loss(logits,target)+.25*hard_pixel_loss(logits,target)
