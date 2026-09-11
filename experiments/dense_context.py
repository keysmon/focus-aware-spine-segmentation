"""Nine ordered input/output planes with efficient 2D convolutions."""
from torch import nn
from segmentation.model import SmallUNet


class DenseContextUNet(SmallUNet):
    def __init__(self, channels=9):
        if channels != 9:
            raise ValueError('Expected nine ordered focal planes')
        super().__init__()
        self.encoders[0][0] = nn.Conv2d(9, 16, 3, padding=1)
        self.output = nn.Conv2d(16, 9, 1)

    def forward_volume(self, x):
        if x.ndim != 4 or x.shape[1] != 9:
            raise ValueError('Expected N9HW ordered focal neighborhoods')
        return super().forward(x)[:, None]

    def forward(self, x):
        return self.forward_volume(x)[:, :, 4]
