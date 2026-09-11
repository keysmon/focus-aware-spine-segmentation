"""Single-variable architecture probe: instance normalization in U-Net blocks."""
from torch import nn
from segmentation.model import SmallUNet


def normalized(block):
    layers = []
    for layer in block:
        layers.append(layer)
        if isinstance(layer, nn.Conv2d):
            layers.append(nn.InstanceNorm2d(layer.out_channels, affine=True))
    return nn.Sequential(*layers)


class NormalizedUNet(SmallUNet):
    def __init__(self, channels=1):
        if channels != 1:
            raise ValueError('Expected one input channel')
        super().__init__()
        self.encoders = nn.ModuleList([normalized(block) for block in self.encoders])
        self.center = normalized(self.center)
        self.decoders = nn.ModuleList([normalized(block) for block in self.decoders])
