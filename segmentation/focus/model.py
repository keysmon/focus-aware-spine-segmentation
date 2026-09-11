"""Context-width adaptation; the historical model remains unchanged."""
import numpy as np
import torch
from torch import nn
from segmentation.model import SmallUNet
from segmentation.focus.data import context


class FocusUNet(SmallUNet):
    def __init__(self, channels):
        if channels not in (1,5):
            raise ValueError('Expected context width 1 or 5')
        super().__init__()
        if channels!=1:
            self.encoders[0][0]=nn.Conv2d(channels,16,3,padding=1)
        self.channels=channels


def predictor(model,device):
    def predict(batch):
        model.eval()
        with torch.no_grad():
            return model(torch.from_numpy(batch).to(device)).sigmoid().cpu().numpy()
    return predict


def load_predictor(path,device):
    checkpoint=torch.load(path,map_location=device,weights_only=True)
    model=FocusUNet(checkpoint['channels']).to(device)
    model.load_state_dict(checkpoint['model'])
    return predictor(model,device)


def predict_stack(images,predict,channels,size=128,stride=64):
    if channels not in (1,5) or not 0<stride<=size:
        raise ValueError('Invalid context width or stride')
    output=[]
    for z in range(len(images)):
        x=context(images,z,channels)
        h,w=x.shape[-2:]
        x=np.pad(x,((0,0),(0,max(0,size-h)),(0,max(0,size-w))),mode='edge')
        def starts(n): return sorted(set([*range(0,n-size+1,stride),n-size]))
        coords=[(r,c) for r in starts(x.shape[-2]) for c in starts(x.shape[-1])]
        sums=np.zeros(x.shape[-2:],np.float32); counts=np.zeros_like(sums)
        for i in range(0,len(coords),4):
            group=coords[i:i+4]
            batch=np.asarray([x[:,r:r+size,c:c+size] for r,c in group],np.float32)
            probs=predict(batch)
            if probs.shape!=(len(group),1,size,size) or not np.isfinite(probs).all() or probs.min()<0 or probs.max()>1:
                raise ValueError('Predictor must return finite N1HW probabilities in [0,1]')
            for (r,c),prob in zip(group,probs[:,0]):
                sums[r:r+size,c:c+size]+=prob
                counts[r:r+size,c:c+size]+=1
        if not counts.all(): raise ValueError('Uncovered reconstruction pixels')
        output.append((sums/counts)[:h,:w])
    return np.asarray(output,np.float32)
