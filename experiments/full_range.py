"""Per-stack min/max scaling without percentile clipping."""
import numpy as np


def normalize_stack(images):
    x=np.asarray(images)
    if x.ndim!=3 or not x.size or not np.isfinite(x).all():
        raise ValueError('Expected a finite nonempty ZHW stack')
    lo,hi=float(x.min()),float(x.max())
    normalized=(x.astype(np.float32)-lo)/(hi-lo) if hi>lo else np.zeros_like(x,dtype=np.float32)
    return normalized.astype(np.float32),[lo,hi]
