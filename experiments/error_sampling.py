"""Fixed training-error mining mixed equally with original candidate pools."""
import numpy as np
from segmentation.focus.data import Sampler


def mix_candidates(original,errors):
    if not len(errors):return original
    if not len(original):return errors
    repeated=np.tile(errors,(int(np.ceil(len(original)/len(errors))),1))[:len(original)]
    return np.concatenate([original,repeated])


class ErrorSampler(Sampler):
    def __init__(self,pairs,probabilities,seed,size=128):
        if len(pairs)!=len(probabilities):raise ValueError('One training prediction per pair required')
        super().__init__(pairs,seed,size)
        for i,((x,y),p) in enumerate(zip(pairs,probabilities)):
            p=np.asarray(p)
            if p.shape!=y.shape or not np.isfinite(p).all() or p.min()<0 or p.max()>1:
                raise ValueError('Expected aligned finite training probabilities in [0,1]')
            for z in range(len(y)):
                misses=np.argwhere(y[z]&(p[z]<.3))
                false_alarms=np.argwhere((~y[z])&(p[z]>=.3))
                self.positive[i][z]=mix_candidates(self.positive[i][z],misses)
                self.negative[i][z]=mix_candidates(self.negative[i][z],false_alarms)
