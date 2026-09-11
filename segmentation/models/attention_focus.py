"""Additive sigmoid skip gates inspired by Attention U-Net (1804.03999)."""
import torch
from torch import nn
from torch.nn import functional as F
from segmentation.focus.model import FocusUNet


class SkipGate(nn.Module):
    def __init__(self, skip_channels, context_channels):
        super().__init__()
        hidden = skip_channels // 2
        self.skip = nn.Conv2d(skip_channels, hidden, 1)
        self.context = nn.Conv2d(context_channels, hidden, 1)
        self.score = nn.Conv2d(hidden, 1, 1)

    def forward(self, skip, context):
        weight = self.score(F.relu(self.skip(skip) + self.context(context))).sigmoid()
        return skip * weight


class AttentionFocusUNet(FocusUNet):
    def __init__(self, channels):
        super().__init__(channels)
        self.gates = nn.ModuleList([SkipGate(64, 128), SkipGate(32, 64), SkipGate(16, 32)])

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.center(x)
        for decoder, gate, skip in zip(self.decoders, self.gates, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = decoder(torch.cat((x, gate(skip, x)), dim=1))
        return self.output(x)
