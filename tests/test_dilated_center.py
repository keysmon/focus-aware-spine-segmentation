import torch
from experiments.dilated_center import DilatedCenterUNet
from segmentation.focus.model import FocusUNet


def test_same_initial_parameters_native_shape_and_gradients():
    torch.manual_seed(435);base=FocusUNet(1)
    torch.manual_seed(435);model=DilatedCenterUNet(1)
    assert list(base.state_dict())==list(model.state_dict())
    for key,value in base.state_dict().items():torch.testing.assert_close(value,model.state_dict()[key])
    x=torch.rand(2,1,63,79);y=model(x)
    assert y.shape==x.shape and torch.isfinite(y).all()
    y.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.center[0].dilation==(2,2) and model.center[2].dilation==(4,4)
    clone=DilatedCenterUNet(1);clone.load_state_dict(model.state_dict(),strict=True)
    torch.testing.assert_close(clone(x),y)
