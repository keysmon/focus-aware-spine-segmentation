"""Eight-group feature normalization in the original small U-Net."""
from torch import nn
from segmentation.focus.model import FocusUNet

class GroupFocusUNet(FocusUNet):
    def __init__(self, channels):
        super().__init__(channels)
        def normalized(block):
            layers=[]
            for layer in block:
                layers.append(layer)
                if isinstance(layer,nn.Conv2d):
                    layers.append(nn.GroupNorm(8,layer.out_channels))
            return nn.Sequential(*layers)
        self.encoders=nn.ModuleList([normalized(b) for b in self.encoders])
        self.center=normalized(self.center)
        self.decoders=nn.ModuleList([normalized(b) for b in self.decoders])
