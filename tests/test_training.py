import numpy as np
import torch
from segmentation.model import SmallUNet
from segmentation.train import load_predictor


def test_model_preserves_shape_and_backpropagates():
    torch.set_num_threads(2)
    model = SmallUNet()
    logits = model(torch.zeros(2, 1, 128, 128))
    assert logits.shape == (2, 1, 128, 128)
    logits.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_checkpoint_roundtrip(tmp_path):
    model = SmallUNet().eval()
    path = tmp_path / 'model.pt'
    torch.save({'model': model.state_dict()}, path)
    x = np.zeros((1, 1, 128, 128), np.float32)
    with torch.no_grad():
        expected = model(torch.from_numpy(x)).sigmoid().numpy()
    np.testing.assert_allclose(load_predictor(path, 'cpu')(x), expected, atol=1e-6)


def test_tiny_fold_writes_usable_checkpoint(tmp_path):
    from segmentation.train import train_fold
    x = np.zeros((1, 128, 128), np.uint8)
    x[:, 32:64, 32:64] = 255
    y = x != 0
    config = dict(seed=435, epochs=1, batches_per_epoch=1, batch_size=1,
                  training_seconds=10, patience=5, learning_rate=0.001,
                  patch_size=128, stride=64, thresholds=[0.3, 0.5, 0.7])
    result = train_fold([(x, y)], (x, y), config, tmp_path/'fold')
    assert result['selected']['epoch'] == 1
    assert result['selected']['threshold'] in config['thresholds']
    assert (tmp_path/'fold'/'history.json').exists()
    assert np.isfinite(load_predictor(tmp_path/'fold'/'best.pt', 'cpu')(x[:, None].astype(np.float32))).all()


def test_missing_checkpoint_rejected(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        load_predictor(tmp_path/'missing.pt', 'cpu')
