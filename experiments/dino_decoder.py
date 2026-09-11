"""Native-image refinement from frozen DINOv2 features sampled at stride seven."""
import torch
from torch import nn
from torch.nn import functional as F


def prepare_input(x):
    if x.ndim!=4 or x.shape[1]!=1 or not torch.isfinite(x).all():
        raise ValueError('Expected finite N1HW normalized grayscale images')
    x=F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)
    h,w=x.shape[-2:]
    x=F.pad(x,(0,(-w)%14,0,(-h)%14),mode='replicate')
    mean=x.new_tensor([.485,.456,.406])[None,:,None,None]
    std=x.new_tensor([.229,.224,.225])[None,:,None,None]
    return (x.repeat(1,3,1,1)-mean)/std


class DenseDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.project=nn.Sequential(nn.Conv2d(384,32,1),nn.ReLU())
        self.refine=nn.Sequential(nn.Conv2d(33,16,3,padding=1),nn.ReLU(),
                                  nn.Conv2d(16,16,3,padding=1),nn.ReLU(),nn.Conv2d(16,1,1))

    def forward(self,features,image):
        h,w=image.shape[-2:]
        if features.ndim!=4 or features.shape[1]!=384 or image.ndim!=4 or image.shape[1]!=1 or features.shape[0]!=image.shape[0]:
            raise ValueError('Expected matching N384hw features and N1HW image')
        fh,fw=features.shape[-2:]
        if h>fh*7 or w>fw*7:raise ValueError('Feature grid does not cover image')
        x=F.interpolate(self.project(features),size=(fh*7,fw*7),mode='bilinear',align_corners=False)
        return self.refine(torch.cat([x[:,:,:h,:w],image],dim=1))
