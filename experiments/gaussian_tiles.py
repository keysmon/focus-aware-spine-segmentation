"""Fixed Gaussian center weighting for overlapping native tiles."""
import numpy as np
from segmentation.focus.data import context

def predict_stack(images,predict,channels,size=128,stride=64):
    if channels not in (1,5) or not 0<stride<=size:
        raise ValueError('Invalid context width or stride')
    coordinates=np.arange(size,dtype=np.float32)-(size-1)/2
    g=np.exp(-.5*(coordinates/(size/8))**2)
    weight=np.maximum(g[:,None]*g[None,:],1e-4)
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
                sums[r:r+size,c:c+size]+=prob*weight
                counts[r:r+size,c:c+size]+=weight
        if not counts.all(): raise ValueError('Uncovered reconstruction pixels')
        output.append((sums/counts)[:h,:w])
    return np.asarray(output,np.float32)
