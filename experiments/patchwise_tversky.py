"""Average per-patch overlap to avoid pooling foreground volume across samples."""
from torch.nn import functional as F


def loss_value(logits,target):
    if logits.shape!=target.shape or logits.ndim!=4 or not logits.numel():
        raise ValueError('Expected matching nonempty NCHW arrays')
    p=logits.sigmoid().flatten(1);y=target.flatten(1)
    tp=(p*y).sum(1);fp=(p*(1-y)).sum(1);fn=((1-p)*y).sum(1)
    overlap=1-(tp+1)/(tp+.3*fp+.7*fn+1)
    return F.binary_cross_entropy_with_logits(logits,target)+overlap.mean()
