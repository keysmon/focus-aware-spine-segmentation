import numpy as np
from segmentation.patches import augment_pair, sample_patch, predict_tiled


def test_tiled_identity_reconstructs_edges():
    for shape in [(2, 137, 151), (1, 17, 23)]:
        x = np.random.default_rng(435).random(shape, dtype=np.float32)
        np.testing.assert_allclose(predict_tiled(x, lambda batch: batch), x, atol=1e-6)


def test_paired_augmentation():
    mask = np.zeros((32, 32), bool); mask[3:12, 7:17] = True
    for seed in range(10):
        image, actual = augment_pair(mask.astype(np.float32), mask, np.random.default_rng(seed))
        np.testing.assert_array_equal(image > 0.5, actual)
        assert actual.dtype == bool


def test_empty_patch_sampling():
    x = np.zeros((2, 32, 32), np.float32)
    a, b = sample_patch(x, x.astype(bool), np.random.default_rng(1))
    assert a.shape == b.shape == (128, 128) and not b.any()
