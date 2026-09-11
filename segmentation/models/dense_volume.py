"""Dense supervision of existing masks across ordered focal neighborhoods."""
import numpy as np
from segmentation.models.ordered_volume import VolumeUNet, VolumeSampler, context


# Both training modes share the same architecture and checkpoint format.
DenseVolumeUNet = VolumeUNet


class DenseVolumeSampler(VolumeSampler):
    def sample(self, channels=9):
        x, _, meta = super().sample(channels)
        y = context(self.pairs[meta['pair']][1], meta['z'], channels)
        h, w = y.shape[-2:]
        y = np.pad(y, ((0, 0), (0, max(0, self.size-h)),
                       (0, max(0, self.size-w))), mode='constant')
        r, c = meta['top'], meta['left']
        y = y[:, r:r+self.size, c:c+self.size]
        y = np.rot90(y, meta['k'], axes=(-2, -1))
        if meta['flip_y']:
            y = np.flip(y, -2)
        if meta['flip_x']:
            y = np.flip(y, -1)
        return x, np.ascontiguousarray(y, dtype=bool), meta
