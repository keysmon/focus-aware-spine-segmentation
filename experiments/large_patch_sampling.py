"""Paired reflection extends narrow training frames for larger crop experiments."""
import numpy as np
from segmentation.focus.data import Sampler as BaseSampler


class Sampler(BaseSampler):
    def __init__(self, pairs, seed, size=256):
        extended=[]
        for images,masks in pairs:
            if images.ndim!=3 or images.shape!=masks.shape:
                raise ValueError('Image/mask shapes must match in ZHW')
            padding=((0,0),(0,max(0,size-images.shape[1])),
                     (0,max(0,size-images.shape[2])))
            extended.append((np.pad(images,padding,mode='reflect'),
                             np.pad(masks,padding,mode='reflect')))
        super().__init__(extended,seed,size)
