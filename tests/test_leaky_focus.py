import torch
from torch import nn
from segmentation.focus.model import FocusUNet
from experiments.leaky_focus import LeakyFocusUNet

def test_negative_gradient_and_checkpoint_contract():
    torch.manual_seed(435);base=FocusUNet(1)
    torch.manual_seed(435);model=LeakyFocusUNet(1)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in base.state_dict().items())
    assert not any(isinstance(m,nn.ReLU) for m in model.modules())
    x=torch.tensor([-2.],requires_grad=True);model.encoders[0][1](x).sum().backward()
    assert torch.allclose(x.grad,torch.tensor([.1]))
    image=torch.randn(2,1,31,43);out=model(image)
    assert out.shape==(2,1,31,43)
    out.sum().backward()
    restored=LeakyFocusUNet(1);restored.load_state_dict(model.state_dict(),strict=True)
    assert torch.equal(out,restored(image))
