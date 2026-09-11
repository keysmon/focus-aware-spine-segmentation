"""Preserve the sampling schedule while omitting gain/offset clipping."""
import numpy as np
from segmentation.focus.data import Sampler, context


class NoPhotometricSampler(Sampler):
    def sample(self, channels):
        _, y, meta = super().sample(channels)
        images, _ = self.pairs[meta['pair']]
        h, w = images.shape[1:]
        x = np.pad(context(images, meta['z'], channels),
                   ((0, 0), (0, max(0, self.size-h)), (0, max(0, self.size-w))),
                   mode='edge')
        top, left = meta['top'], meta['left']
        x = x[:, top:top+self.size, left:left+self.size]
        x = np.rot90(x, meta['k'], axes=(-2, -1))
        if meta['flip_y']:
            x = np.flip(x, -2)
        if meta['flip_x']:
            x = np.flip(x, -1)
        return np.ascontiguousarray(x, dtype=np.float32), y, meta
