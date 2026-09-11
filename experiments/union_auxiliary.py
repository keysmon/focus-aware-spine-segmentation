"""Training-only weak structure supervision; focus output stays independent."""
import numpy as np
import torch
from torch import nn
from segmentation.model import SmallUNet
from segmentation.focus.data import Sampler
from experiments.tversky import loss_value as primary_loss
from segmentation.train import loss_value as auxiliary_loss


class UnionUNet(SmallUNet):
    def __init__(self,channels=1):
        if channels!=1:raise ValueError('Expected one image channel')
        super().__init__()
        original=self.output
        self.output=nn.Conv2d(16,2,1)
        with torch.no_grad():
            self.output.weight[:1].copy_(original.weight)
            self.output.bias[:1].copy_(original.bias)

    def forward_heads(self,x):
        return super().forward(x)

    def forward(self,x):
        return self.forward_heads(x)[:,:1]


class UnionSampler(Sampler):
    def __init__(self,pairs,seed,size=128):
        super().__init__(pairs,seed,size)
        self.unions=[y.any(axis=0) for _,y in pairs]

    def sample(self,channels=1):
        x,y,meta=super().sample(channels)
        u=self.unions[meta['pair']]
        h,w=u.shape
        u=np.pad(u,((0,max(0,self.size-h)),(0,max(0,self.size-w))))
        r,c=meta['top'],meta['left'];u=u[r:r+self.size,c:c+self.size]
        u=np.rot90(u,meta['k'])
        if meta['flip_y']:u=np.flip(u,-2)
        if meta['flip_x']:u=np.flip(u,-1)
        return x,np.ascontiguousarray(np.stack([y,u]),dtype=bool),meta


def loss_value(logits,target,weight=.25):
    if logits.ndim!=4 or logits.shape!=target.shape or logits.shape[1]!=2:
        raise ValueError('Expected matching N2HW logits and targets')
    return primary_loss(logits[:,:1],target[:,:1])+weight*auxiliary_loss(logits[:,1:],target[:,1:])
