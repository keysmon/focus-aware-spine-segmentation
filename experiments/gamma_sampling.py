"""Training-only monotonic intensity augmentation; masks are unchanged."""
import json
import numpy as np
from segmentation.focus.data import Sampler


def gamma_transform(image,gamma):
    x=np.asarray(image,dtype=np.float32)
    if not np.isfinite(gamma) or gamma<=0:raise ValueError('Expected positive finite gamma')
    if not x.size or not np.isfinite(x).all() or x.min()<0 or x.max()>1:
        raise ValueError('Expected finite normalized image')
    return np.power(x,gamma).astype(np.float32)


class GammaSampler(Sampler):
    def __init__(self,pairs,seed,size=128):
        super().__init__(pairs,seed,size)
        self.gamma_rng=np.random.default_rng(seed+1)

    def sample(self,channels=1):
        x,y,meta=super().sample(channels)
        if self.gamma_rng.random()>=.5:return x,y,meta
        gamma=float(self.gamma_rng.uniform(.7,1.) if self.gamma_rng.random()<.5 else self.gamma_rng.uniform(1.,1.5))
        x=gamma_transform(x,gamma);meta['gamma']=gamma
        self._digest.update(json.dumps(meta,sort_keys=True).encode())
        return x,y,meta
