"""Capacity probe retaining the original U-Net operations and depth."""
from torch import nn
from segmentation.model import SmallUNet, block


class WideUNet(SmallUNet):
    def __init__(self, channels=1):
        if channels != 1:
            raise ValueError('Expected one input channel')
        nn.Module.__init__(self)
        self.encoders = nn.ModuleList([block(1, 32), block(32, 64), block(64, 128)])
        self.center = block(128, 256)
        self.decoders = nn.ModuleList([block(384, 128), block(192, 64), block(96, 32)])
        self.output = nn.Conv2d(32, 1, 1)
