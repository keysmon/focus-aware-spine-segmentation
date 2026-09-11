"""BCE plus focal Tversky with author implementation's exponent 0.75."""
import torch


def loss_value(logits, target):
    probability = logits.sigmoid()
    tp = (probability * target).sum()
    fp = (probability * (1 - target)).sum()
    fn = ((1 - probability) * target).sum()
    error = 1 - (tp + 1) / (tp + .3 * fp + .7 * fn + 1)
    # Avoid the singular derivative at exact zero in saturated float32 inputs.
    focal = error.clamp_min(1e-7).pow(.75)
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, target) + focal
