"""Permit pretrained encoder population statistics to adapt on training data."""
from torch import nn
from experiments.resnet_unet import ResNetUNet


class AdaptiveResNetUNet(ResNetUNet):
    def train(self,mode=True):
        super().train(mode)
        for layer in self.encoder.modules():
            if isinstance(layer,nn.BatchNorm2d):
                layer.train(mode)
        return self
