"""Refine four frozen transformer feature levels at native image resolution."""
import torch
from torch import nn
from torch.nn import functional as F
from experiments.dino_decoder import DenseDecoder


class MultiLayerDecoder(DenseDecoder):
    def __init__(self):
        super().__init__()
        self.project=nn.Sequential(nn.Conv2d(1536,32,1),nn.ReLU())
        self.feature_dropout=nn.Dropout2d(.5)

    def forward(self,features,image):
        if features.ndim!=4 or features.shape[1]!=1536 or image.ndim!=4 or image.shape[1]!=1 or features.shape[0]!=image.shape[0]:
            raise ValueError('Expected N1536hw features and matching N1HW image')
        h,w=image.shape[-2:];fh,fw=features.shape[-2:]
        if h>fh*7 or w>fw*7:raise ValueError('Feature grid does not cover image')
        x=self.project(self.feature_dropout(features))
        x=F.interpolate(x,size=(fh*7,fw*7),mode='bilinear',align_corners=False)
        return self.refine(torch.cat([x[:,:,:h,:w],image],dim=1))
