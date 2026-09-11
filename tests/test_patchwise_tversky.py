import torch
from experiments.patchwise_tversky import loss_value
from experiments.tversky import loss_value as pooled


def test_patchwise_equals_average_of_independent_losses():
    torch.manual_seed(435)
    logits=torch.randn(3,1,8,8,requires_grad=True)
    target=torch.zeros_like(logits);target[0,:,:2,:2]=1;target[1,:,:6,:6]=1
    expected=torch.stack([pooled(logits[i:i+1],target[i:i+1]) for i in range(3)]).mean()
    torch.testing.assert_close(loss_value(logits,target),expected)
    assert abs(float((expected-pooled(logits,target)).detach()))>1e-3
    expected.backward();assert torch.isfinite(logits.grad).all()


def test_patchwise_handles_empty_full_and_single_pixel():
    for label in (0.,1.):
        logits=torch.zeros(2,1,1,1,requires_grad=True)
        target=torch.full_like(logits,label)
        loss=loss_value(logits,target);loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
        assert ((logits.grad>0) if label==0 else (logits.grad<0)).all()
