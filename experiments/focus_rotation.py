"""Rotation-only augmentation with higher-fidelity image interpolation."""
import json
import cv2
import numpy as np
from segmentation.focus.data import Sampler


def rotate_pair(image,mask,top,left,size,angle):
    center=(left+(size-1)/2,top+(size-1)/2)
    matrix=cv2.getRotationMatrix2D(center,angle,1.)
    matrix[:,2]+=np.array([(size-1)/2-center[0],(size-1)/2-center[1]])
    x=cv2.warpAffine(image,matrix,(size,size),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_REFLECT_101)
    y=cv2.warpAffine(mask.astype(np.uint8),matrix,(size,size),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_REFLECT_101)
    return np.clip(x,0,1).astype(np.float32),y.astype(bool)


class RotationSampler(Sampler):
    def __init__(self,pairs,seed,size=128):
        super().__init__(pairs,seed,size)
        self.rotation_rng=np.random.default_rng(seed+1)

    def sample(self,channels=1):
        if channels!=1:raise ValueError('Expected one image channel')
        x,y,meta=super().sample(1)
        if self.rotation_rng.random()>=.5:return x,y,meta
        images,masks=self.pairs[meta['pair']]
        x,y=rotate_pair(images[meta['z']],masks[meta['z']],meta['top'],meta['left'],self.size,meta['k']*90+45)
        if meta['flip_y']:x,y=np.flip(x,-2),np.flip(y,-2)
        if meta['flip_x']:x,y=np.flip(x,-1),np.flip(y,-1)
        x=np.clip(x*meta['gain']+meta['offset'],0,1)
        meta['extra_angle']=45
        self._digest.update(json.dumps(meta,sort_keys=True).encode())
        return np.ascontiguousarray(x[None],dtype=np.float32),np.ascontiguousarray(y),meta
