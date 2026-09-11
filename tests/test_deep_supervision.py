import torch
from segmentation.model import SmallUNet
from experiments.deep_supervision import DeepSupervisedUNet, loss_value


def test_deep_supervision_preserves_initial_final_output_and_native_geometry():
    torch.manual_seed(435);base=SmallUNet()
    torch.manual_seed(435);model=DeepSupervisedUNet()
    x=torch.randn(2,1,35,43)
    torch.testing.assert_close(base(x),model(x),rtol=0,atol=0)
    heads=model.forward_heads(x)
    assert [tuple(v.shape) for v in heads]==[(2,1,35,43),(2,1,17,21),(2,1,8,10)]
    clone=DeepSupervisedUNet();clone.load_state_dict(model.state_dict(),strict=True)
    torch.testing.assert_close(clone(x),model(x),rtol=0,atol=0)


def test_auxiliary_losses_reach_all_heads_and_encoder_with_empty_examples():
    model=DeepSupervisedUNet();x=torch.randn(2,1,32,40)
    y=torch.zeros_like(x);y[0,0,3:29,12:14]=1
    loss=loss_value(model.forward_heads(x),y);loss.backward()
    assert torch.isfinite(loss)
    for layer in [model.output,*model.auxiliary,model.encoders[0][0]]:
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all() and layer.weight.grad.abs().sum()>0
