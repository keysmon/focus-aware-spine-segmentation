"""Microscopy-pretrained Cellpose backbone adapted to binary focus segmentation."""
from pathlib import Path
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from experiments.vendor.cellpose_resnet import CPnet

WEIGHT_SHA256='2dc3087a8abd7da46d1ab0ddd5824639933cc3ff63b382af3fa1939a392db93c'
WEIGHTS=Path(__file__).resolve().parents[1]/'runs/cellpose-assets/cyto3'


class CellposeFocus(nn.Module):
    def __init__(self,channels=1,pretrained=True):
        super().__init__()
        if channels!=1:raise ValueError('Expected one image channel')
        self.net=CPnet([2,32,64,128,256],3,3,mkldnn=False)
        if pretrained:
            if hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()!=WEIGHT_SHA256:
                raise ValueError('Unexpected Cellpose checkpoint hash')
            self.net.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True),strict=True)
        self.net.output[2]=nn.Conv2d(32,1,1)
        self.train(True)

    def train(self,mode=True):
        super().train(mode)
        # Keep microscopy-pretrained running statistics; affine weights can adapt.
        for layer in self.modules():
            if isinstance(layer,nn.BatchNorm2d):layer.eval()
        return self

    def forward(self,x):
        if x.ndim!=4 or x.shape[1]!=1:raise ValueError('Expected N1HW input')
        h,w=x.shape[-2:]
        x=F.pad(x,(0,(-w)%8,0,(-h)%8),mode='replicate')
        # Cellpose grayscale convention: image channel plus absent nuclear channel.
        x=torch.cat([x,torch.zeros_like(x)],dim=1)
        logits,_,_=self.net(x)
        return logits[...,:h,:w]
