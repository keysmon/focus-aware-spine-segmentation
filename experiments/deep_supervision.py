"""SmallUNet with two training-only coarse mask prediction heads."""
import torch
from torch import nn
from torch.nn import functional as F
from segmentation.model import SmallUNet
from experiments.tversky import loss_value as primary_loss


class DeepSupervisedUNet(SmallUNet):
    def __init__(self,channels=1):
        if channels!=1:raise ValueError('Expected one image channel')
        super().__init__()
        self.auxiliary=nn.ModuleList([nn.Conv2d(64,1,1),nn.Conv2d(32,1,1)])

    def forward_heads(self,x):
        skips=[]
        for encoder in self.encoders:
            x=encoder(x);skips.append(x);x=F.max_pool2d(x,2)
        x=self.center(x);coarse=[]
        for i,(decoder,skip) in enumerate(zip(self.decoders,reversed(skips))):
            x=F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False)
            x=decoder(torch.cat((x,skip),dim=1))
            if i<2:coarse.append(self.auxiliary[i](x))
        return self.output(x),coarse[1],coarse[0]

    def forward(self,x):
        # Normal inference uses only the original full-resolution path.
        return super().forward(x)


def loss_value(heads,target):
    if len(heads)!=3 or target.ndim!=4 or heads[0].shape!=target.shape:
        raise ValueError('Expected full, half, quarter N1HW logits and matching target')
    weights=(1.,.5,.25)
    losses=[]
    for logits,weight in zip(heads,weights):
        if logits.shape[:2]!=target.shape[:2]:raise ValueError('Head batch/channel mismatch')
        # Area averaging retains thin foreground as fractional coarse occupancy.
        coarse=F.adaptive_avg_pool2d(target,logits.shape[-2:])
        losses.append(weight*primary_loss(logits,coarse))
    return sum(losses)/sum(weights)
