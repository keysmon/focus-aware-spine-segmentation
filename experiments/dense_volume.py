"""Dense supervision of existing masks across ordered focal neighborhoods."""
import numpy as np
import torch
from torch.nn import functional as F
from experiments.ordered_volume import VolumeUNet, VolumeSampler, context


class DenseVolumeUNet(VolumeUNet):
    def forward_volume(self,x):
        if x.ndim!=4 or x.shape[1]!=9:
            raise ValueError('Expected N9HW ordered focal neighborhoods')
        x=x[:,None];skips=[]
        for encoder in self.encoders:
            x=encoder(x);skips.append(x)
            x=F.max_pool3d(x,(1,2,2))
        x=self.center(x)
        for decoder,skip in zip(self.decoders,reversed(skips)):
            # Resize only XY; preserve the ordered focal axis exactly.
            n,c,z,h,w=x.shape
            x=x.permute(0,2,1,3,4).reshape(n*z,c,h,w)
            x=F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False)
            x=x.reshape(n,z,c,*skip.shape[-2:]).permute(0,2,1,3,4)
            x=decoder(torch.cat([x,skip],dim=1))
        return self.output(x)


    def forward(self, x):
        return self.forward_volume(x)[:, :, 4]


class DenseVolumeSampler(VolumeSampler):
    def sample(self, channels=9):
        x, _, meta = super().sample(channels)
        y = context(self.pairs[meta['pair']][1], meta['z'], channels)
        h, w = y.shape[-2:]
        y = np.pad(y, ((0, 0), (0, max(0, self.size-h)),
                       (0, max(0, self.size-w))), mode='constant')
        r, c = meta['top'], meta['left']
        y = y[:, r:r+self.size, c:c+self.size]
        y = np.rot90(y, meta['k'], axes=(-2, -1))
        if meta['flip_y']:
            y = np.flip(y, -2)
        if meta['flip_x']:
            y = np.flip(y, -1)
        return x, np.ascontiguousarray(y, dtype=bool), meta
