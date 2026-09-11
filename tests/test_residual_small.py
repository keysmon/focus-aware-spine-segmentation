import torch
from experiments.residual_small import ResidualSmallUNet, ResidualBlock
from segmentation.focus.model import FocusUNet


def test_zero_residual_preserves_first_features_and_gradient():
    block = ResidualBlock(torch.nn.Sequential(torch.nn.Conv2d(1, 3, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(3, 3, 3, padding=1), torch.nn.ReLU()))
    with torch.no_grad():
        block.layers[2].weight.zero_(); block.layers[2].bias.zero_()
    x = torch.randn(2, 1, 13, 17, requires_grad=True)
    expected = block.layers[1](block.layers[0](x))
    torch.testing.assert_close(block(x), expected)
    block(x).sum().backward()
    assert x.grad.abs().sum() > 0


def test_native_shape_checkpoint_and_same_parameter_budget():
    torch.manual_seed(435); original = FocusUNet(1)
    torch.manual_seed(435); model = ResidualSmallUNet(1)
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in original.parameters())
    for a, b in zip(original.parameters(), model.parameters()):
        torch.testing.assert_close(a, b)
    x = torch.rand(2, 1, 31, 43)
    y = model(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    clone = ResidualSmallUNet(1); clone.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(clone(x), y)
    y.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
