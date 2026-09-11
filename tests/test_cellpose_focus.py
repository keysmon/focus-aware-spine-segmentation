import torch
from experiments.cellpose_focus import CellposeFocus


def test_cellpose_binary_head_preserves_native_geometry_and_gradients():
    torch.set_num_threads(2)
    model=CellposeFocus(pretrained=False);model.train()
    x=torch.rand(1,1,35,43,requires_grad=True)
    result=model(x)
    assert result.shape==x.shape and torch.isfinite(result).all()
    result.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert model.net.output[2].weight.grad.abs().sum()>0
    assert all(not m.training for m in model.modules() if isinstance(m,torch.nn.BatchNorm2d))


def test_cellpose_binary_checkpoint_roundtrip():
    model=CellposeFocus(pretrained=False).eval()
    clone=CellposeFocus(pretrained=False).eval();clone.load_state_dict(model.state_dict(),strict=True)
    x=torch.rand(1,1,32,40)
    with torch.no_grad():torch.testing.assert_close(model(x),clone(x),atol=0,rtol=0)
