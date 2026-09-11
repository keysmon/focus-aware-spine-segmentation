"""Learned per-plane segmentation conditioned on one complete focal sweep."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from segmentation.model import SmallUNet
from segmentation.focus.data import Sampler


class SweepUNet(SmallUNet):
    """The leading dimension is Z for one stack, never unrelated batch items."""
    def __init__(self):
        super().__init__()
        self.fusions = nn.ModuleList([nn.Conv2d(2*c, c, 1) for c in (16, 32, 64, 128)])

    def fuse(self, x, layer):
        pooled = x.amax(dim=0, keepdim=True).expand_as(x)
        return F.relu(layer(torch.cat([x, pooled], dim=1)))

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 1 or x.shape[0] < 1:
            raise ValueError('Expected one nonempty Z1HW sweep')
        skips = []
        for encoder, fusion in zip(self.encoders, self.fusions):
            x = self.fuse(encoder(x), fusion)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.fuse(self.center(x), self.fusions[-1])
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = decoder(torch.cat([x, skip], dim=1))
        return self.output(x)


class SweepSampler(Sampler):
    def sample(self, channels=1):
        _, _, meta = super().sample(1)
        x, y = self.pairs[meta['pair']]
        h, w = x.shape[-2:]
        padding = ((0, 0), (0, max(0, self.size-h)), (0, max(0, self.size-w)))
        x = np.pad(x, padding, mode='edge')
        y = np.pad(y, padding)
        r, c = meta['top'], meta['left']
        x, y = x[:, r:r+self.size, c:c+self.size], y[:, r:r+self.size, c:c+self.size]
        x, y = np.rot90(x, meta['k'], axes=(-2, -1)), np.rot90(y, meta['k'], axes=(-2, -1))
        if meta['flip_y']:
            x, y = np.flip(x, -2), np.flip(y, -2)
        if meta['flip_x']:
            x, y = np.flip(x, -1), np.flip(y, -1)
        x = np.clip(x*meta['gain']+meta['offset'], 0, 1)
        return np.ascontiguousarray(x[:, None], dtype=np.float32), np.ascontiguousarray(y[:, None], dtype=np.float32), meta


def predict_sweep(images, predict, size=128, stride=64):
    x = np.asarray(images, np.float32)
    if x.ndim != 3 or not x.size or not np.isfinite(x).all() or not 0 < stride <= size:
        raise ValueError('Expected finite ZHW images and valid tiling settings')
    z, h, w = x.shape
    x = np.pad(x, ((0, 0), (0, max(0, size-h)), (0, max(0, size-w))), mode='edge')
    def starts(n):
        return sorted(set([*range(0, n-size+1, stride), n-size]))
    total = np.zeros_like(x)
    count = np.zeros(x.shape[-2:], np.float32)
    for r in starts(x.shape[-2]):
        for c in starts(x.shape[-1]):
            probability = predict(x[:, None, r:r+size, c:c+size])
            if probability.shape != (z, 1, size, size) or not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1:
                raise ValueError('Expected finite Z1HW probabilities')
            total[:, r:r+size, c:c+size] += probability[:, 0]
            count[r:r+size, c:c+size] += 1
    if not count.all():
        raise ValueError('Uncovered reconstruction')
    return (total/count[None])[:, :h, :w]
