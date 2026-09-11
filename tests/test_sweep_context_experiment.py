import numpy as np
import torch
from experiments.sweep_context import SweepUNet, SweepSampler, predict_sweep


def test_sweep_model_is_plane_permutation_equivariant():
    torch.set_num_threads(2)
    torch.manual_seed(1)
    model = SweepUNet().eval()
    x = torch.rand(3, 1, 24, 32)
    order = torch.tensor([2, 0, 1])
    with torch.no_grad():
        a = model(x)
        b = model(x[order])
    assert a.shape == x.shape and torch.isfinite(a).all()
    torch.testing.assert_close(b, a[order], atol=1e-6, rtol=1e-5)


def test_sweep_geometry_preserves_all_plane_labels():
    y = np.zeros((3, 20, 24), bool)
    y[0, 2:10, 4:13] = True
    y[1, 9:16, 12:19] = True
    sampler = SweepSampler([(y.astype(np.float32), y)], 435, 16)
    for _ in range(12):
        x, target, _ = sampler.sample()
        assert x.shape == target.shape == (3, 1, 16, 16)
        np.testing.assert_array_equal(x > .5, target != 0)


def test_sweep_inference_never_mixes_spatial_tiles_as_planes():
    x = np.random.default_rng(4).random((3, 19, 25)).astype(np.float32)
    def predict(batch):
        assert batch.shape == (3, 1, 16, 16)
        return batch
    actual = predict_sweep(x, predict, size=16, stride=8)
    np.testing.assert_allclose(actual, x, atol=1e-7)
