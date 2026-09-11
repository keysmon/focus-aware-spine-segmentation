"""Experimental U-Net with image, local contrast and relative sweep contrast."""
import numpy as np
import torch
from torch import nn
from scipy.ndimage import uniform_filter
from segmentation.model import SmallUNet
from segmentation.focus.data import Sampler


def cue_maps(images):
    """Normalized ZHW image -> Z2HW cues. No labels or cross-stack statistics."""
    x = np.asarray(images, dtype=np.float32)
    if x.ndim != 3 or not x.size or not np.isfinite(x).all():
        raise ValueError('Expected finite nonempty ZHW images')
    if x.min() < 0 or x.max() > 1:
        raise ValueError('Expected normalized intensities')
    mean = uniform_filter(x, (1, 9, 9))
    variance = np.maximum(uniform_filter(x*x, (1, 9, 9)) - mean*mean, 0)
    std = np.sqrt(variance)
    absolute = np.clip(2*std, 0, 1)
    relative = std / (std.max(axis=0, keepdims=True) + 1e-5)
    return np.stack([absolute, relative], axis=1).astype(np.float32)


def transform_cues(cues, meta, size):
    """Apply exactly the sampler's geometry to two auxiliary channels."""
    v = cues[meta['z']]
    h, w = v.shape[-2:]
    v = np.pad(v, ((0, 0), (0, max(0, size-h)), (0, max(0, size-w))), mode='edge')
    r, c = meta['top'], meta['left']
    v = v[:, r:r+size, c:c+size]
    v = np.rot90(v, meta['k'], axes=(-2, -1))
    if meta['flip_y']:
        v = np.flip(v, -2)
    if meta['flip_x']:
        v = np.flip(v, -1)
    return np.ascontiguousarray(v)


class CueSampler(Sampler):
    def __init__(self, pairs, seed, size=128):
        super().__init__(pairs, seed, size)
        self.cues = [cue_maps(x) for x, _ in pairs]

    def sample(self, channels=3):
        image, mask, meta = super().sample(1)
        extra = transform_cues(self.cues[meta['pair']], meta, self.size)
        # Cues describe the original normalized stack and share all geometry.
        # Photometric augmentation affects the image channel only.
        return np.concatenate([image, extra]), mask, meta


class CueUNet(SmallUNet):
    def __init__(self, channels=3):
        if channels != 3:
            raise ValueError('Expected three channels')
        super().__init__()
        self.encoders[0][0] = nn.Conv2d(3, 16, 3, padding=1)


def predict_stack(images, predict, channels=3, size=128, stride=64):
    if channels != 3 or not 0 < stride <= size:
        raise ValueError('Invalid cue inference settings')
    extra = cue_maps(images)
    inputs = np.concatenate([images[:, None], extra], axis=1)
    output = []
    for x in inputs:
        h, w = x.shape[-2:]
        x = np.pad(x, ((0, 0), (0, max(0, size-h)), (0, max(0, size-w))), mode='edge')
        def starts(n):
            return sorted(set([*range(0, n-size+1, stride), n-size]))
        coords = [(r, c) for r in starts(x.shape[-2]) for c in starts(x.shape[-1])]
        sums = np.zeros(x.shape[-2:], np.float32)
        counts = np.zeros_like(sums)
        for i in range(0, len(coords), 4):
            group = coords[i:i+4]
            batch = np.asarray([x[:, r:r+size, c:c+size] for r, c in group], np.float32)
            probs = predict(batch)
            if probs.shape != (len(group), 1, size, size) or not np.isfinite(probs).all() or probs.min() < 0 or probs.max() > 1:
                raise ValueError('Expected finite N1HW probabilities')
            for (r, c), p in zip(group, probs[:, 0]):
                sums[r:r+size, c:c+size] += p
                counts[r:r+size, c:c+size] += 1
        if not counts.all():
            raise ValueError('Uncovered pixels')
        output.append((sums/counts)[:h, :w])
    return np.asarray(output)
