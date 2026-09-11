"""Original crop RNG schedule, with raw image and mask orientation."""
import numpy as np
from segmentation.focus.data import Sampler,context


class UnaugmentedSampler(Sampler):
    def sample(self,channels):
        _,_,m=super().sample(channels)
        images,masks=self.pairs[m['pair']];h,w=images.shape[1:]
        x=np.pad(context(images,m['z'],channels),((0,0),(0,max(0,self.size-h)),(0,max(0,self.size-w))),mode='edge')
        y=np.pad(masks[m['z']],((0,max(0,self.size-h)),(0,max(0,self.size-w))))
        top,left=m['top'],m['left']
        return np.ascontiguousarray(x[:,top:top+self.size,left:left+self.size],dtype=np.float32),np.ascontiguousarray(y[top:top+self.size,left:left+self.size],dtype=bool),m
