import torch
from experiments.centerline_loss import centerline_loss,loss_value


def test_centerline_loss_penalizes_gaps_without_forcing_disconnected_labels_together():
    y=torch.zeros(1,1,24,24);y[:,:,3:21,10:13]=1
    gap=y.clone();gap[:,:,10:14]=0
    assert centerline_loss(y,y)==0
    assert centerline_loss(gap,y)>0
    assert centerline_loss(gap,gap)==0


def test_centerline_loss_handles_empty_targets_and_has_finite_gradients():
    y=torch.zeros(2,1,24,24);y[0,0,3:21,10:13]=1
    assert centerline_loss(torch.zeros_like(y),torch.zeros_like(y))==0
    x=torch.randn(y.shape,requires_grad=True);loss=loss_value(x,y);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
