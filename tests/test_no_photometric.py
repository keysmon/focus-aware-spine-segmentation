import numpy as np
from experiments.no_photometric import NoPhotometricSampler
from segmentation.focus.data import Sampler


def test_matched_schedule_masks_and_only_photometric_difference():
    rng = np.random.default_rng(5)
    images = rng.uniform(.01, .99, (4, 13, 11)).astype(np.float32)
    masks = images > .7
    for channels in (1, 5):
        base = Sampler([(images, masks)], 435, 16)
        candidate = NoPhotometricSampler([(images, masks)], 435, 16)
        for _ in range(40):
            bx, by, bm = base.sample(channels)
            x, y, m = candidate.sample(channels)
            assert bm == m
            np.testing.assert_array_equal(y, by)
            np.testing.assert_array_equal(bx, np.clip(x*m['gain']+m['offset'], 0, 1))
            assert x.shape == (channels, 16, 16)
            assert x.flags.c_contiguous and x.dtype == np.float32
            assert x.min() > 0 and x.max() < 1
        assert candidate.schedule_hash() == base.schedule_hash()
        assert candidate.counts == base.counts
