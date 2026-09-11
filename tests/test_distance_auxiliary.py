import numpy as np
import torch
from experiments.distance_auxiliary import signed_distance, DistanceSampler, DistanceUNet, loss_value
from segmentation.focus.data import Sampler


def test_distance_geometry_empty_and_sampler_alignment():
    y=np.zeros((21,23),bool);y[5:16,6:17]=True
    d=signed_distance(y)
    assert d[10,11]>d[5,6]>0 and d[0,0]<0
    np.testing.assert_array_equal(d>0,y)
    assert (signed_distance(np.zeros_like(y))==-1).all()
    pairs=[(np.stack([y,y]).astype(np.float32),np.stack([y,y]))]
    a=DistanceSampler(pairs,435,24);b=Sampler(pairs,435,24)
    for _ in range(20):
        x,t,m=a.sample();ox,oy,om=b.sample(1)
        assert m==om
        np.testing.assert_array_equal(x,ox)
        np.testing.assert_array_equal(t[0],oy)
        np.testing.assert_array_equal(t[1]>0,oy)
        assert t.dtype==np.float32 and np.isfinite(t).all()


def test_distance_heads_gradient_and_primary_inference():
    torch.set_num_threads(2)
    model=DistanceUNet();x=torch.rand(2,1,17,19);p=model.forward_heads(x)
    torch.testing.assert_close(model(x),p[:,:1])
    t=torch.zeros_like(p);t[:,1]=-1
    loss=loss_value(p,t);loss.backward()
    assert torch.isfinite(loss)
    assert (model.output.weight.grad.flatten(1).abs().sum(1)>0).all()
