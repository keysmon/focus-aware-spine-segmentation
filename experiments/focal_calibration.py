"""Image-only relative focal confidence; this is not a calibrated probability."""
import numpy as np


def relative_confidence(probabilities):
    p=np.asarray(probabilities,dtype=np.float32)
    if p.ndim!=3 or not p.size or not np.isfinite(p).all() or p.min()<0 or p.max()>1:
        raise ValueError('Expected finite nonempty ZHW probabilities in [0,1]')
    peak=p.max(axis=0,keepdims=True)
    return p/(.25+.75*peak)
