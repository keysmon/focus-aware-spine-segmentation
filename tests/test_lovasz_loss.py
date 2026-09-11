import itertools
import torch
from experiments.lovasz_loss import lovasz_hinge, loss_value


def test_lovasz_matches_discrete_jaccard_errors_at_binary_margins():
    target = torch.tensor([1., 1., 0., 0.])
    for bits in itertools.product((0., 1.), repeat=4):
        prediction = torch.tensor(bits)
        logits = prediction * 2 - 1
        intersection = (prediction * target).sum()
        union = ((prediction + target) > 0).sum()
        expected = 2 * (1 - intersection / union)
        torch.testing.assert_close(lovasz_hinge(logits, target), expected)


def test_lovasz_empty_full_and_wrong_predictions_have_finite_gradients():
    for target in (torch.zeros(8), torch.ones(8), torch.tensor([0.,1.] * 4)):
        logits = ((1 - target) * 4 - 2).requires_grad_()
        loss = loss_value(logits, target)
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
        assert (logits.grad[target == 0] > 0).all()
        assert (logits.grad[target == 1] < 0).all()
        correct = target * 4 - 2
        assert lovasz_hinge(correct, target) == 0
