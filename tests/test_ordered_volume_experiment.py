import numpy as np
import torch
from experiments.ordered_volume import VolumeUNet, VolumeSampler, predict_stack
from segmentation.focus.data import Sampler


def test_volume_sampler_center_matches_existing_augmentation():
    rng=np.random.default_rng(1)
    x=rng.random((12,40,43),dtype=np.float32);y=x>.8
    base=Sampler([(x,y)],seed=435,size=32)
    volume=VolumeSampler([(x,y)],seed=435,size=32)
    for _ in range(12):
        a,b,meta=base.sample(1);u,v,newmeta=volume.sample(9)
        assert u.shape==(9,32,32)
        assert meta==newmeta and np.array_equal(a[0],u[4])
        assert np.array_equal(b,v)


def test_volume_model_uses_neighbors_and_preserves_center_shape():
    torch.set_num_threads(2);torch.manual_seed(435)
    model=VolumeUNet(9)
    x=torch.rand(2,9,33,41,requires_grad=True)
    y=model(x)
    assert y.shape==(2,1,33,41) and torch.isfinite(y).all()
    y[0].sum().backward()
    assert x.grad[0,3].abs().sum()>0
    assert x.grad[1].abs().sum()==0


def test_volume_reconstruction_has_no_slice_offset_or_edge_loss():
    x=np.random.default_rng(4).random((3,19,23),dtype=np.float32)
    seen=[]
    def center(batch):
        seen.append(batch.shape)
        return batch[:,4:5]
    p=predict_stack(x,center,9,size=16,stride=7)
    np.testing.assert_allclose(p,x,atol=1e-7)
    assert all(shape[1]==9 for shape in seen)
