import torch
from experiments.wide_unet import WideUNet


def test_wide_model_backprop_and_checkpoint_roundtrip(tmp_path):
    torch.set_num_threads(2)
    model = WideUNet()
    x = torch.rand(1, 1, 32, 40)
    y = model(x)
    assert y.shape == x.shape
    y.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    path = tmp_path/'model.pt'
    torch.save(model.state_dict(), path)
    restored = WideUNet()
    restored.load_state_dict(torch.load(path, weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(restored(x), model(x), atol=0, rtol=0)
