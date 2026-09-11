import torch
from experiments.hard_pixel_loss import hard_pixel_loss,loss_value


def test_hard_pixel_term_targets_confident_errors_in_both_classes():
    x=torch.tensor([[[[-4.,4.],[4.,-4.]]]],requires_grad=True)
    y=torch.tensor([[[[1.,1.],[0.,0.]]]])
    loss=hard_pixel_loss(x,y);loss.backward()
    assert x.grad[0,0,0,0]<0 and x.grad[0,0,1,0]>0
    assert x.grad[0,0,0,1]==0 and x.grad[0,0,1,1]==0
    assert torch.isfinite(loss)


def test_hard_pixel_term_handles_single_class_batches_and_small_masks():
    for value in (0.,1.):
        x=torch.zeros((1,1,1,1),requires_grad=True);y=torch.full_like(x,value)
        loss=loss_value(x,y);loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
