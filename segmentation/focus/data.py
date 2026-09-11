"""Stack-consistent inputs and training-only matched sampling."""
import hashlib
import json
import cv2
import numpy as np


def normalize_stack(images):
    x = np.asarray(images)
    if x.ndim != 3 or not x.size or not np.isfinite(x).all():
        raise ValueError('Expected a finite nonempty ZHW stack')
    lo, hi = np.percentile(x, (1, 99))
    normalized = np.clip((x.astype(np.float32)-lo)/(hi-lo), 0, 1) if hi > lo else np.zeros_like(x,dtype=np.float32)
    return normalized.astype(np.float32), [float(lo),float(hi)]


def context(images, z, channels):
    if channels not in (1,5) or not 0 <= z < len(images):
        raise ValueError('Expected context width 1 or 5 and a valid center index')
    indices = np.clip(np.arange(z-channels//2,z+channels//2+1),0,len(images)-1)
    return images[indices]


class Sampler:
    """RNG consumption is independent of channel count and network behavior."""
    def __init__(self, pairs, seed, size=128):
        if not pairs or size <= 0:
            raise ValueError('Nonempty training pairs and positive size required')
        self.pairs = pairs
        self.rng = np.random.default_rng(seed)
        self.size = size
        self.positive, self.negative = [], []
        self.counts = dict(foreground=0, difficult_negative=0, uniform=0, fallback=0)
        self._digest = hashlib.sha256()
        for images,masks in pairs:
            if images.shape != masks.shape or images.ndim != 3:
                raise ValueError('Image/mask shapes must match')
            positive,negative = [],[]
            for frame,mask in zip(images,masks):
                positive.append(np.argwhere(mask))
                gx=cv2.Sobel(frame,cv2.CV_32F,1,0,ksize=3)
                gy=cv2.Sobel(frame,cv2.CV_32F,0,1,ksize=3)
                energy=cv2.GaussianBlur(gx*gx+gy*gy,(5,5),0)
                values=energy[~mask]
                threshold=float(np.percentile(values,90)) if len(values) else np.inf
                negative.append(np.argwhere((~mask)&(energy>=threshold)&(energy>0)))
            self.positive.append(positive)
            self.negative.append(negative)

    def schedule_hash(self):
        return self._digest.hexdigest()

    def sample(self, channels):
        rng=self.rng
        pair=int(rng.integers(len(self.pairs)))
        images,masks=self.pairs[pair]
        category=int(rng.integers(3))
        self.counts[('foreground','difficult_negative','uniform')[category]] += 1
        candidates=(self.positive if category==0 else self.negative)[pair] if category<2 else None
        eligible=[i for i,c in enumerate(candidates) if len(c)] if candidates is not None else []
        if eligible:
            z=int(rng.choice(eligible))
            cy,cx=map(int,candidates[z][int(rng.integers(len(candidates[z])))])
        else:
            if category<2: self.counts['fallback'] += 1
            z=int(rng.integers(len(images)))
            cy,cx=int(rng.integers(images.shape[1])),int(rng.integers(images.shape[2]))
        size=self.size
        h,w=images.shape[1:]
        top=int(np.clip(cy-size//2,0,max(0,h-size)))
        left=int(np.clip(cx-size//2,0,max(0,w-size)))
        x=np.pad(context(images,z,channels),((0,0),(0,max(0,size-h)),(0,max(0,size-w))),mode='edge')
        y=np.pad(masks[z],((0,max(0,size-h)),(0,max(0,size-w))))
        x=x[:,top:top+size,left:left+size]
        y=y[top:top+size,left:left+size]
        k=int(rng.integers(4)); flip_y=bool(rng.integers(2)); flip_x=bool(rng.integers(2))
        gain=float(rng.uniform(.9,1.1)); offset=float(rng.uniform(-.05,.05))
        x=np.rot90(x,k,axes=(-2,-1)); y=np.rot90(y,k)
        if flip_y: x=np.flip(x,-2); y=np.flip(y,-2)
        if flip_x: x=np.flip(x,-1); y=np.flip(y,-1)
        x=np.clip(x*gain+offset,0,1)
        meta=dict(pair=pair,z=z,top=top,left=left,category=category,k=k,flip_y=flip_y,flip_x=flip_x,gain=gain,offset=offset)
        self._digest.update(json.dumps(meta,sort_keys=True).encode())
        return np.ascontiguousarray(x,dtype=np.float32),np.ascontiguousarray(y,dtype=bool),meta
