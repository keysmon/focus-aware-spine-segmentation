"""ResNet-18-compatible encoder and native-resolution segmentation decoder.

Architecture follows He et al.'s residual network; state names match the
official PyTorch ResNet-18 checkpoint. No network downloads occur here.
"""
import hashlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, inputs, outputs, stride=1):
        super().__init__()
        self.conv1=nn.Conv2d(inputs,outputs,3,stride,1,bias=False)
        self.bn1=nn.BatchNorm2d(outputs)
        self.relu=nn.ReLU()
        self.conv2=nn.Conv2d(outputs,outputs,3,1,1,bias=False)
        self.bn2=nn.BatchNorm2d(outputs)
        self.downsample=nn.Sequential(nn.Conv2d(inputs,outputs,1,stride,bias=False),nn.BatchNorm2d(outputs)) if inputs!=outputs or stride!=1 else None

    def forward(self,x):
        shortcut=x if self.downsample is None else self.downsample(x)
        value=self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(value))+shortcut)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,64,7,2,3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU()
        self.maxpool=nn.MaxPool2d(3,2,1)
        self.layer1=nn.Sequential(ResidualBlock(64,64),ResidualBlock(64,64))
        self.layer2=nn.Sequential(ResidualBlock(64,128,2),ResidualBlock(128,128))
        self.layer3=nn.Sequential(ResidualBlock(128,256,2),ResidualBlock(256,256))
        self.layer4=nn.Sequential(ResidualBlock(256,512,2),ResidualBlock(512,512))
        for layer in self.modules():
            if isinstance(layer,nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight,mode='fan_out',nonlinearity='relu')

    def forward(self,x):
        stem=self.relu(self.bn1(self.conv1(x)))
        a=self.layer1(self.maxpool(stem))
        b=self.layer2(a)
        c=self.layer3(b)
        d=self.layer4(c)
        return stem,a,b,c,d


def decoder_block(inputs,outputs):
    return nn.Sequential(nn.Conv2d(inputs,outputs,3,padding=1),nn.GroupNorm(8,outputs),nn.ReLU(),
                         nn.Conv2d(outputs,outputs,3,padding=1),nn.GroupNorm(8,outputs),nn.ReLU())


class ResNetUNet(nn.Module):
    def __init__(self,channels=1,weights=None):
        super().__init__()
        if channels!=1:
            raise ValueError('Expected one grayscale image channel')
        self.encoder=Encoder()
        self.decoders=nn.ModuleList([decoder_block(768,256),decoder_block(384,128),
                                     decoder_block(192,64),decoder_block(128,32),decoder_block(33,16)])
        self.output=nn.Conv2d(16,1,1)
        self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])
        if weights is not None:
            path=Path(weights)
            if not hashlib.sha256(path.read_bytes()).hexdigest().startswith('f37072fd'):
                raise ValueError('Unexpected official ResNet-18 weight hash')
            state=torch.load(path,map_location='cpu',weights_only=True)
            self.encoder.load_state_dict({k:v for k,v in state.items() if not k.startswith('fc.')},strict=True)
        self.train()

    def train(self,mode=True):
        super().train(mode)
        # Preserve encoder population statistics during small-data fine-tuning.
        for layer in self.encoder.modules():
            if isinstance(layer,nn.BatchNorm2d):
                layer.eval()
        return self

    def forward(self,x):
        if x.ndim!=4 or x.shape[1]!=1:
            raise ValueError('Expected N1HW images')
        features=self.encoder((x.repeat(1,3,1,1)-self.mean)/self.std)
        value=features[-1]
        for decoder,skip in zip(self.decoders,(*reversed(features[:-1]),x)):
            value=F.interpolate(value,size=skip.shape[-2:],mode='bilinear',align_corners=False)
            value=decoder(torch.cat([value,skip],dim=1))
        return self.output(value)
