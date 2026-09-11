"""Auxiliary signed distance regression from existing in-focus masks."""
import numpy as np
from scipy.ndimage import distance_transform_edt
from torch.nn import functional as F
from experiments.union_auxiliary import UnionUNet as DistanceUNet
from experiments.tversky import loss_value as primary_loss
from segmentation.focus.data import Sampler


def signed_distance(mask):
    if mask.dtype != bool or mask.ndim != 2:
        raise ValueError('Expected a 2D boolean mask')
    if not mask.any():
        return np.full(mask.shape, -1, np.float32)
    if mask.all():
        return np.ones(mask.shape, np.float32)
    return np.clip((distance_transform_edt(mask)-distance_transform_edt(~mask))/16.,-1,1).astype(np.float32)


class DistanceSampler(Sampler):
    def __init__(self,pairs,seed,size=128):
        super().__init__(pairs,seed,size)
        self.distances=[np.stack([signed_distance(y) for y in masks]) for _,masks in pairs]

    def sample(self,channels=1):
        x,y,m=super().sample(channels)
        d=self.distances[m['pair']][m['z']]
        h,w=d.shape
        d=np.pad(d,((0,max(0,self.size-h)),(0,max(0,self.size-w))),constant_values=-1)
        r,c=m['top'],m['left'];d=d[r:r+self.size,c:c+self.size]
        d=np.rot90(d,m['k'])
        if m['flip_y']:d=np.flip(d,-2)
        if m['flip_x']:d=np.flip(d,-1)
        return x,np.ascontiguousarray(np.stack([y,d]),dtype=np.float32),m


def loss_value(logits,target):
    return primary_loss(logits[:,:1],target[:,:1])+.25*F.smooth_l1_loss(logits[:,1:].tanh(),target[:,1:])
