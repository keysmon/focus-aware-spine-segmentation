"""Average downsampling with otherwise unchanged small U-Net weights."""
import torch
from torch.nn import functional as F
from segmentation.focus.model import FocusUNet

class AverageFocusUNet(FocusUNet):
    @staticmethod
    def pool(x):
        return F.avg_pool2d(x,2)

    def forward(self,x):
        skips=[]
        for encoder in self.encoders:
            x=encoder(x);skips.append(x);x=self.pool(x)
        x=self.center(x)
        for decoder,skip in zip(self.decoders,reversed(skips)):
            x=F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False)
            x=decoder(torch.cat((x,skip),dim=1))
        return self.output(x)
