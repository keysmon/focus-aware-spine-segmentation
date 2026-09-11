"""2D soft centerline overlap following Shit et al., CVPR 2021 clDice.

Reference equations: https://arxiv.org/abs/2003.07311.
The main focus segmentation objective and reference masks remain unchanged.
"""
import torch
from torch.nn import functional as F
from experiments.tversky import loss_value as primary_loss


def erode(x):
    vertical=-F.max_pool2d(-x,(3,1),stride=1,padding=(1,0))
    horizontal=-F.max_pool2d(-x,(1,3),stride=1,padding=(0,1))
    return torch.minimum(vertical,horizontal)


def skeleton(x,iterations=10):
    if x.ndim!=4:raise ValueError('Expected NCHW masks')
    result=torch.zeros_like(x)
    for _ in range(iterations+1):
        smaller=erode(x)
        opened=F.max_pool2d(smaller,3,stride=1,padding=1)
        detail=(x-opened).relu()
        result=result+(detail-result*detail).relu()
        x=smaller
    return result


def centerline_loss(probability,target):
    if probability.shape!=target.shape:raise ValueError('Mask shapes must match')
    predicted_center=skeleton(probability)
    target_center=skeleton(target)
    precision=((predicted_center*target).sum()+1)/(predicted_center.sum()+1)
    recall=((target_center*probability).sum()+1)/(target_center.sum()+1)
    return 1-2*precision*recall/(precision+recall)


def loss_value(logits,target):
    return primary_loss(logits,target)+.25*centerline_loss(logits.sigmoid(),target)
