"""Fixed negative-slope activations, preserving convolution initialization."""
from torch import nn
from segmentation.focus.model import FocusUNet

class LeakyFocusUNet(FocusUNet):
    def __init__(self, channels):
        super().__init__(channels)
        for block in [*self.encoders,self.center,*self.decoders]:
            for index,layer in enumerate(block):
                if isinstance(layer,nn.ReLU):
                    block[index]=nn.LeakyReLU(negative_slope=.1)
