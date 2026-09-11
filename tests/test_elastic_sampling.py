import numpy as np
from experiments.elastic_sampling import warp_pair, ElasticSampler
from segmentation.focus.data import Sampler


def test_warp_preserves_pair_alignment_binary_mask_and_inputs():
    y=np.zeros((16,19),bool);y[4:10,6:12]=True
    x=np.stack([y,y]).astype(np.float32);before=x.copy()
    displacement=np.zeros((2,16,19),np.float32);displacement[1]=2
    a,b=warp_pair(x,y,displacement)
    np.testing.assert_array_equal(a[0],b)
    np.testing.assert_array_equal(a[1],b)
    np.testing.assert_array_equal(b[:,4:10],y[:,6:12])
    np.testing.assert_array_equal(x,before)
    assert b.dtype==bool and a.dtype==np.float32


def test_elastic_reproducible_preserves_base_draws_and_contracts():
    x=np.random.default_rng(1).random((3,40,45),dtype=np.float32);y=x>.8
    a=ElasticSampler([(x,y)],435,32);b=ElasticSampler([(x,y)],435,32)
    base=Sampler([(x,y)],435,32);changed=0
    for _ in range(20):
        u,v,m=a.sample();p,q,n=b.sample();raw,mask,meta=base.sample(1)
        np.testing.assert_array_equal(u,p);np.testing.assert_array_equal(v,q)
        for k in meta:assert m[k]==meta[k]
        changed+=not np.array_equal(u,raw)
        assert u.shape==raw.shape and v.shape==mask.shape and v.dtype==bool
        assert np.isfinite(u).all() and 0<=u.min()<=u.max()<=1
    assert 0<changed<20
