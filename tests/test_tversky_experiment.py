import torch
from experiments.tversky import loss_value


def test_missing_foreground_costs_more_than_equal_false_positive():
    fn = loss_value(torch.tensor([8., -8., -8.]), torch.tensor([1., 1., 0.]))
    fp = loss_value(torch.tensor([8., 8., -8.]), torch.tensor([1., 0., 0.]))
    assert fn > fp


def test_empty_targets_and_extreme_logits_have_finite_gradients():
    for target in (torch.zeros(4), torch.ones(4)):
        logits = torch.tensor([-100., -5., 5., 100.], requires_grad=True)
        loss = loss_value(logits, target)
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
