"""Ordered focal neighborhoods with XY-only pooling and center-plane outputs."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from segmentation.focus.data import Sampler


def context(images,z,channels=9):
    if channels!=9 or not 0<=z<len(images):
        raise ValueError('Expected nine ordered planes and valid center index')
    return images[np.clip(np.arange(z-4,z+5),0,len(images)-1)]


def block(inputs,outputs):
    return nn.Sequential(nn.Conv3d(inputs,outputs,3,padding=1),nn.ReLU(),
                         nn.Conv3d(outputs,outputs,3,padding=1),nn.ReLU())


class VolumeUNet(nn.Module):
    def __init__(self,channels=9):
        super().__init__()
        if channels!=9: raise ValueError('Expected nine focal planes')
        self.encoders=nn.ModuleList([block(1,8),block(8,16),block(16,32)])
        self.center=block(32,64)
        self.decoders=nn.ModuleList([block(96,32),block(48,16),block(24,8)])
        self.output=nn.Conv3d(8,1,1)

    def forward(self,x):
        if x.ndim!=4 or x.shape[1]!=9:
            raise ValueError('Expected N9HW ordered focal neighborhoods')
        x=x[:,None];skips=[]
        for encoder in self.encoders:
            x=encoder(x);skips.append(x)
            x=F.max_pool3d(x,(1,2,2))
        x=self.center(x)
        for decoder,skip in zip(self.decoders,reversed(skips)):
            # Resize only XY; preserve the ordered focal axis exactly.
            n,c,z,h,w=x.shape
            x=x.permute(0,2,1,3,4).reshape(n*z,c,h,w)
            x=F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False)
            x=x.reshape(n,z,c,*skip.shape[-2:]).permute(0,2,1,3,4)
            x=decoder(torch.cat([x,skip],dim=1))
        return self.output(x)[:,:,4]


class VolumeSampler(Sampler):
    def sample(self,channels=9):
        if channels!=9: raise ValueError('Expected nine focal planes')
        _,y,meta=super().sample(1)
        x=context(self.pairs[meta['pair']][0],meta['z'])
        h,w=x.shape[-2:]
        x=np.pad(x,((0,0),(0,max(0,self.size-h)),(0,max(0,self.size-w))),mode='edge')
        r,c=meta['top'],meta['left'];x=x[:,r:r+self.size,c:c+self.size]
        x=np.rot90(x,meta['k'],axes=(-2,-1))
        if meta['flip_y']:x=np.flip(x,-2)
        if meta['flip_x']:x=np.flip(x,-1)
        x=np.clip(x*meta['gain']+meta['offset'],0,1)
        return np.ascontiguousarray(x,dtype=np.float32),y,meta


def predict_stack(images,predict,channels,size=128,stride=64):
    if channels != 9 or not 0<stride<=size:
        raise ValueError('Invalid context width or stride')
    output=[]
    for z in range(len(images)):
        x=context(images,z,channels)
        h,w=x.shape[-2:]
        x=np.pad(x,((0,0),(0,max(0,size-h)),(0,max(0,size-w))),mode='edge')
        def starts(n): return sorted(set([*range(0,n-size+1,stride),n-size]))
        coords=[(r,c) for r in starts(x.shape[-2]) for c in starts(x.shape[-1])]
        sums=np.zeros(x.shape[-2:],np.float32); counts=np.zeros_like(sums)
        for i in range(0,len(coords),4):
            group=coords[i:i+4]
            batch=np.asarray([x[:,r:r+size,c:c+size] for r,c in group],np.float32)
            probs=predict(batch)
            if probs.shape!=(len(group),1,size,size) or not np.isfinite(probs).all() or probs.min()<0 or probs.max()>1:
                raise ValueError('Predictor must return finite N1HW probabilities in [0,1]')
            for (r,c),prob in zip(group,probs[:,0]):
                sums[r:r+size,c:c+size]+=prob
                counts[r:r+size,c:c+size]+=1
        if not counts.all(): raise ValueError('Uncovered reconstruction pixels')
        output.append((sums/counts)[:h,:w])
    return np.asarray(output,np.float32)
