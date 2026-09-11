"""Fixed image-only local contrast preprocessing; reference masks are unchanged."""
import math
import cv2
import numpy as np
from segmentation.focus.data import Sampler


def enhance_stack(images):
    """Blend normalized ZHW input with per-plane CLAHE using ~64-pixel tiles.

    Enhancement is computed on entire planes before patch extraction, identically
    for training and inference. No labels or neighboring planes are consulted.
    """
    x=np.asarray(images,dtype=np.float32)
    if x.ndim!=3 or not x.size or not np.isfinite(x).all():
        raise ValueError('Expected finite nonempty ZHW images')
    if x.min()<0 or x.max()>1:
        raise ValueError('Expected normalized intensities in [0, 1]')
    grid=(math.ceil(x.shape[2]/64),math.ceil(x.shape[1]/64))
    clahe=cv2.createCLAHE(clipLimit=2.,tileGridSize=grid)
    enhanced=np.stack([clahe.apply(np.rint(plane*255).astype(np.uint8)) for plane in x]).astype(np.float32)/255
    return (0.5*x+0.5*enhanced).astype(np.float32)


class ContrastSampler(Sampler):
    def __init__(self,pairs,seed,size=128):
        # Preserve original-image hard-negative candidates and RNG consumption.
        super().__init__(pairs,seed,size)
        self.pairs=[(enhance_stack(x),y) for x,y in pairs]
