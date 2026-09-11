"""Binary Lovasz hinge over a batch, following Berman et al., CVPR 2018.

https://arxiv.org/abs/1705.08790
Sort margin errors and weight them by increments of the Jaccard set loss.
"""
import torch
from torch.nn import functional as F


def lovasz_hinge(logits, target):
    if logits.shape != target.shape or logits.numel() == 0:
        raise ValueError('Expected matching nonempty logits and targets')
    scores = logits.reshape(-1)
    labels = target.reshape(-1).to(scores.dtype)
    errors = 1 - scores * (2 * labels - 1)
    sorted_errors, order = torch.sort(errors, descending=True)
    ordered_labels = labels[order]
    positives = labels.sum()
    intersection = positives - ordered_labels.cumsum(0)
    union = positives + (1 - ordered_labels).cumsum(0)
    jaccard = 1 - intersection / union
    increments = torch.cat((jaccard[:1], jaccard[1:] - jaccard[:-1]))
    return (F.relu(sorted_errors) * increments).sum()


def loss_value(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target) + lovasz_hinge(logits, target)
