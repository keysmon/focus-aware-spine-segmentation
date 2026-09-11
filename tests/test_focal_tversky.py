import torch
from experiments.focal_tversky import loss_value


def test_focal_matches_hand_calculation_and_gradient_direction():
    logits=torch.zeros(1,1,1,2,requires_grad=True)
    target=torch.tensor([[[[1.,0.]]]])
    # TP=.5, FP=.5, FN=.5; smoothed index=1.5/2.
    expected=torch.log(torch.tensor(2.))+torch.tensor(.25).pow(.75)
    torch.testing.assert_close(loss_value(logits,target),expected)
    loss_value(logits,target).backward()
    assert logits.grad[0,0,0,0]<0 and logits.grad[0,0,0,1]>0


def test_focal_saturated_predictions_have_finite_gradients():
    for label in (0.,1.):
        for value in (-100.,100.):
            x=torch.full((2,1,3,3),value,requires_grad=True)
            loss=loss_value(x,torch.full_like(x,label));loss.backward()
            assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
