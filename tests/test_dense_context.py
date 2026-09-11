import pytest
import torch
from experiments.dense_context import DenseContextUNet
from experiments.tversky import loss_value


def test_dense_context_shape_gradients_and_checkpoint():
    torch.set_num_threads(2)
    model = DenseContextUNet()
    x = torch.rand(2, 9, 17, 19)
    logits = model.forward_volume(x)
    assert logits.shape == (2, 1, 9, 17, 19)
    torch.testing.assert_close(model(x), logits[:, :, 4])
    target = torch.zeros_like(logits)
    target[:, :, :, 3:8, 4:10] = 1
    loss = loss_value(logits, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert (model.output.weight.grad.flatten(1).abs().sum(1) > 0).all()
    assert model.encoders[0][0].weight.grad.abs().sum() > 0
    clone = DenseContextUNet()
    clone.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(clone(x), model(x))


def test_dense_context_rejects_wrong_context():
    with pytest.raises(ValueError):
        DenseContextUNet(5)
    with pytest.raises(ValueError):
        DenseContextUNet()(torch.rand(1, 5, 16, 16))
