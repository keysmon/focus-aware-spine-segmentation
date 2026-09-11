import numpy as np
import torch
from experiments.dense_volume import DenseVolumeUNet, DenseVolumeSampler
from experiments.ordered_volume import VolumeUNet, VolumeSampler


def test_dense_output_preserves_center_and_trains_neighbor_planes():
    torch.set_num_threads(2)
    torch.manual_seed(435)
    old = VolumeUNet()
    model = DenseVolumeUNet()
    model.load_state_dict(old.state_dict(), strict=True)
    x = torch.rand(1, 9, 17, 19)
    dense = model.forward_volume(x)
    assert dense.shape == (1, 1, 9, 17, 19)
    torch.testing.assert_close(dense[:, :, 4], old(x))
    torch.testing.assert_close(model(x), old(x))
    dense.retain_grad()
    target = torch.rand_like(dense)
    torch.nn.functional.binary_cross_entropy_with_logits(dense, target).backward()
    assert all(dense.grad[:, :, z].abs().sum() > 0 for z in range(9))
    assert model.output.weight.grad.abs().sum() > 0


def test_dense_masks_follow_exact_source_planes_and_augmentation():
    masks = np.zeros((5, 19, 21), bool)
    for z in range(5):
        masks[z, z + 2:z + 5, 3:9] = True
    images = masks.astype(np.float32)
    old = VolumeSampler([(images, masks)], 435, 24)
    new = DenseVolumeSampler([(images, masks)], 435, 24)
    for _ in range(30):
        ox, oy, om = old.sample()
        x, y, meta = new.sample()
        np.testing.assert_array_equal(x, ox)
        np.testing.assert_array_equal(y[4], oy)
        assert meta == om and y.dtype == bool
        indices = np.clip(np.arange(meta['z']-4, meta['z']+5), 0, 4)
        expected = np.pad(masks[indices], ((0,0),(0,5),(0,3)))
        expected = np.rot90(expected, meta['k'], axes=(-2,-1))
        if meta['flip_y']: expected = np.flip(expected, -2)
        if meta['flip_x']: expected = np.flip(expected, -1)
        np.testing.assert_array_equal(y, expected)
