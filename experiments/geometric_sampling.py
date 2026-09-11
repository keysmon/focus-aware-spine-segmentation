"""Explicit paired continuous-angle and scale augmentation for existing stacks."""
import json
import cv2
import numpy as np
from segmentation.focus.data import Sampler


def warp_pair(image, mask, top, left, size, angle, scale):
    center=(left+(size-1)/2,top+(size-1)/2)
    matrix=cv2.getRotationMatrix2D(center,angle,scale)
    matrix[:,2]+=np.array([(size-1)/2-center[0],(size-1)/2-center[1]])
    x=cv2.warpAffine(image,matrix,(size,size),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    y=cv2.warpAffine(mask.astype(np.uint8),matrix,(size,size),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_REFLECT_101)
    return x,y.astype(bool)


class GeometricSampler(Sampler):
    def __init__(self,pairs,seed,size=128):
        super().__init__(pairs,seed,size)
        self.geometry_rng=np.random.default_rng(seed+1)

    def sample(self,channels=1):
        if channels!=1:
            raise ValueError('Expected one image channel')
        _,_,meta=super().sample(1)
        images,masks=self.pairs[meta['pair']]
        angle=float(meta['k']*90+self.geometry_rng.uniform(-45,45))
        scale=float(self.geometry_rng.uniform(.7,1.4))
        x,y=warp_pair(images[meta['z']],masks[meta['z']],meta['top'],meta['left'],self.size,angle,scale)
        if meta['flip_y']:
            x,y=np.flip(x,-2),np.flip(y,-2)
        if meta['flip_x']:
            x,y=np.flip(x,-1),np.flip(y,-1)
        x=np.clip(x*meta['gain']+meta['offset'],0,1)
        meta.update(angle=angle,scale=scale)
        self._digest.update(json.dumps(meta,sort_keys=True).encode())
        return np.ascontiguousarray(x[None],dtype=np.float32),np.ascontiguousarray(y),meta
