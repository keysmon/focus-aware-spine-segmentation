import torch
from torch import nn
from experiments.group_focus import GroupFocusUNet

def test_groupnorm_batch_independence_and_reload():
    torch.manual_seed(435);model=GroupFocusUNet(1)
    assert sum(isinstance(m,nn.GroupNorm) for m in model.modules())==14
    x=torch.randn(1,1,31,43)
    alone=model(x);paired=model(torch.cat([x,torch.randn_like(x)*10]))[:1]
    assert torch.allclose(alone,paired,atol=2e-5)
    assert alone.shape==x.shape
    alone.sum().backward()
    assert all(p.grad is not None for p in model.parameters())
    restored=GroupFocusUNet(1);restored.load_state_dict(model.state_dict(),strict=True)
    assert torch.equal(alone,restored(x))
