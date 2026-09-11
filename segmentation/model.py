"""Small 2D U-Net; returns logits without changing spatial dimensions."""
import torch
from torch import nn
from torch.nn import functional as F


def block(inputs, outputs):
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(outputs, outputs, 3, padding=1), nn.ReLU())


class SmallUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([block(1, 16), block(16, 32), block(32, 64)])
        self.center = block(64, 128)
        self.decoders = nn.ModuleList([block(192, 64), block(96, 32), block(48, 16)])
        self.output = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            x = encoder(x); skips.append(x); x = F.max_pool2d(x, 2)
        x = self.center(x)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = decoder(torch.cat((x, skip), dim=1))
        return self.output(x)
