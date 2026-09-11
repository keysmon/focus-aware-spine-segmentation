"""Strengthen hard-positive BCE gradients while retaining Tversky overlap."""
import torch


def loss_value(logits,target):
    probability=logits.sigmoid()
    tp=(probability*target).sum()
    fp=(probability*(1-target)).sum()
    fn=((1-probability)*target).sum()
    overlap=1-(tp+1)/(tp+.3*fp+.7*fn+1)
    bce=torch.nn.functional.binary_cross_entropy_with_logits(logits,target,pos_weight=logits.new_tensor(5.))
    return bce+overlap
