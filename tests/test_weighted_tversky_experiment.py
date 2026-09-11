import torch
from experiments.tversky import loss_value as original
from experiments.weighted_tversky import loss_value


def test_weighted_loss_increases_hard_positive_gradient():
    a=torch.tensor([-8.,-8.],requires_grad=True)
    b=a.detach().clone().requires_grad_()
    target=torch.tensor([1.,0.])
    original(a,target).backward()
    loss_value(b,target).backward()
    assert b.grad[0]<a.grad[0]<0
    torch.testing.assert_close(b.grad[1],a.grad[1])


def test_weighted_loss_finite_for_empty_target():
    x=torch.tensor([-100.,100.],requires_grad=True)
    loss=loss_value(x,torch.zeros(2))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
