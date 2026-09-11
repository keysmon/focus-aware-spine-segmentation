import torch
from segmentation.focus.model import FocusUNet
from experiments.bottleneck_dropout import BottleneckDropoutUNet

def test_dropout_preserves_eval_and_changes_training():
    torch.manual_seed(435)
    base=FocusUNet(1)
    torch.manual_seed(435)
    model=BottleneckDropoutUNet(1)
    x=torch.randn(2,1,31,43)
    base.eval();model.eval()
    assert torch.equal(base(x),model(x))
    model.train()
    assert not torch.equal(model(x),model(x))
    model(x).sum().backward()
    assert model.center[0].weight.grad is not None
    restored=BottleneckDropoutUNet(1)
    restored.load_state_dict(model.state_dict(),strict=True)
    restored.eval();model.eval()
    assert torch.equal(restored(x),model(x))
