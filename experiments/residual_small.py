"""Add an identity shortcut around each block's second convolution."""
from torch import nn
from segmentation.focus.model import FocusUNet


class ResidualBlock(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, x):
        h = self.layers[1](self.layers[0](x))
        return self.layers[3](self.layers[2](h) + h)


class ResidualSmallUNet(FocusUNet):
    def __init__(self, channels):
        super().__init__(channels)
        self.encoders = nn.ModuleList(ResidualBlock(b) for b in self.encoders)
        self.center = ResidualBlock(self.center)
        self.decoders = nn.ModuleList(ResidualBlock(b) for b in self.decoders)
