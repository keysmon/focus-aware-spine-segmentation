import numpy as np
import pytest
from experiments.focus_cues import cue_maps, CueSampler, predict_stack


def test_cues_are_finite_and_follow_plane_permutations():
    x = np.random.default_rng(1).random((3, 12, 17)).astype(np.float32)
    a = cue_maps(x)
    assert a.shape == (3, 2, 12, 17)
    assert np.isfinite(a).all() and a.min() >= 0 and a.max() <= 1
    np.testing.assert_allclose(cue_maps(x[::-1]), a[::-1])
    assert not cue_maps(np.zeros_like(x)).any()
    with pytest.raises(ValueError):
        cue_maps(x * 2)


def test_auxiliary_channels_follow_label_geometry():
    y = np.zeros((2, 16, 16), bool)
    y[:, 2:8, 4:11] = True
    sampler = CueSampler([(y.astype(np.float32), y)], 435, 16)
    # An artificial cue equal to the mask checks exact geometry, not correlation.
    sampler.cues = [np.repeat(y[:, None], 2, axis=1).astype(np.float32)]
    for _ in range(12):
        x, target, _ = sampler.sample()
        np.testing.assert_array_equal(x[1] != 0, target)
        np.testing.assert_array_equal(x[2] != 0, target)


def test_reconstruction_retains_original_dimensions_and_values():
    x = np.random.default_rng(2).random((2, 15, 23)).astype(np.float32)
    actual = predict_stack(x, lambda batch: batch[:, :1], size=16, stride=8)
    np.testing.assert_allclose(actual, x, atol=1e-7)
