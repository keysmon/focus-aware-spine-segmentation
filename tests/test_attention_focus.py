import torch
from segmentation.models.attention_focus import AttentionFocusUNet, SkipGate
from segmentation.focus.model import FocusUNet


def test_gate_attenuates_features_and_receives_both_gradients():
    torch.manual_seed(435)
    gate=SkipGate(4,8)
    x=torch.randn(2,4,9,11,requires_grad=True);g=torch.randn(2,8,9,11,requires_grad=True)
    out=gate(x,g)
    assert out.shape==x.shape and (out.abs()<=x.abs()).all()
    out.square().sum().backward()
    assert x.grad.abs().sum()>0 and g.grad.abs().sum()>0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in gate.parameters())


def test_attention_native_shape_initial_weights_and_reload():
    torch.manual_seed(435);base=FocusUNet(1)
    torch.manual_seed(435);model=AttentionFocusUNet(1)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in base.state_dict().items())
    x=torch.randn(2,1,31,43);y=model(x)
    assert y.shape==x.shape
    y.sum().backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    other=AttentionFocusUNet(1);other.load_state_dict(model.state_dict(),strict=True)
    torch.testing.assert_close(y,other(x),rtol=0,atol=0)


def test_attention_five_slice_context_has_gradients_and_strict_reload():
    torch.manual_seed(435)
    model=AttentionFocusUNet(5)
    x=torch.randn(2,5,31,43,requires_grad=True)
    y=model(x)
    assert y.shape==(2,1,31,43)
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert (x.grad.abs().sum((0,2,3))>0).all()
    restored=AttentionFocusUNet(5)
    restored.load_state_dict(model.state_dict(),strict=True)
    torch.testing.assert_close(y,restored(x),rtol=0,atol=0)
