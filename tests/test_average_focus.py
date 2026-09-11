import torch
from segmentation.focus.model import FocusUNet
from experiments.average_focus import AverageFocusUNet

def test_average_pool_and_model_contract():
    x=torch.tensor([[[[1.,3.],[5.,7.]]]])
    assert AverageFocusUNet.pool(x).item()==4.
    torch.manual_seed(435);base=FocusUNet(1)
    torch.manual_seed(435);model=AverageFocusUNet(1)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in base.state_dict().items())
    image=torch.randn(2,1,31,43);out=model(image)
    assert out.shape==image.shape
    out.sum().backward()
    assert all(p.grad is not None for p in model.parameters())
    other=AverageFocusUNet(1);other.load_state_dict(model.state_dict(),strict=True)
    assert torch.equal(out,other(image))
