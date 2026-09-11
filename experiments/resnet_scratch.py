"""ResNet segmentation configured for learning encoder statistics from scratch."""
from torch import nn
from experiments.resnet_unet import ResNetUNet


class ScratchResNetUNet(ResNetUNet):
    def __init__(self,channels=1,weights=None):
        if weights is not None:
            raise ValueError('This configuration is strictly from scratch')
        super().__init__(channels=channels,weights=None)

    def train(self,mode=True):
        super().train(mode)
        for layer in self.encoder.modules():
            if isinstance(layer,nn.BatchNorm2d):
                layer.train(mode)
        return self
