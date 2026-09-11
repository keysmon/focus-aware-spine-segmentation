"""Channel dropout in the bottleneck; evaluation matches the base architecture."""
from torch import nn
from segmentation.focus.model import FocusUNet

class BottleneckDropoutUNet(FocusUNet):
    def __init__(self, channels):
        super().__init__(channels)
        self.center.append(nn.Dropout2d(0.2))
