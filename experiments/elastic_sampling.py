"""Mild training-only paired elastic deformation with independent random draws."""
import json
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from segmentation.focus.data import Sampler


def warp_pair(image, mask, displacement):
    if image.ndim != 3 or mask.dtype != bool or image.shape[1:] != mask.shape:
        raise ValueError('Expected CHW image and matching boolean HW mask')
    if displacement.shape != (2, *mask.shape) or not np.isfinite(displacement).all():
        raise ValueError('Expected finite 2HW displacement')
    coordinates = np.indices(mask.shape, dtype=np.float32) + displacement
    x = np.stack([map_coordinates(c, coordinates, order=1, mode='reflect') for c in image])
    y = map_coordinates(mask.astype(np.uint8), coordinates, order=0, mode='reflect') > 0
    return np.ascontiguousarray(x, dtype=np.float32), np.ascontiguousarray(y)


class ElasticSampler(Sampler):
    def __init__(self, pairs, seed, size=128):
        super().__init__(pairs, seed, size)
        self.elastic_rng = np.random.default_rng(seed + 1)

    def sample(self, channels=1):
        x, y, meta = super().sample(channels)
        if self.elastic_rng.random() >= .5:
            return x, y, meta
        noise = self.elastic_rng.standard_normal((2, *y.shape)).astype(np.float32)
        displacement = gaussian_filter(noise, sigma=(0, 8, 8), mode='reflect')
        rms = np.sqrt(np.mean(displacement ** 2))
        displacement *= 2 / max(float(rms), 1e-8)
        displacement = np.clip(displacement, -4, 4)
        x, y = warp_pair(x, y, displacement)
        meta['elastic'] = 'sigma8-rms2-cap4'
        self._digest.update(displacement.tobytes())
        self._digest.update(json.dumps(meta, sort_keys=True).encode())
        return x, y, meta
