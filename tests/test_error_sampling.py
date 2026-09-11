import numpy as np
import pytest
from segmentation.focus.data import Sampler
from experiments.error_sampling import ErrorSampler


def test_error_sampling_emphasizes_misses_without_changing_labels_or_images():
    rng=np.random.default_rng(7);x=rng.random((1,24,24),dtype=np.float32)
    y=np.zeros_like(x,dtype=bool);y[:,4:20,4:20]=True
    p=y.astype(np.float32)*.9;p[0,10,10]=.01;p[0,1,1]=.99
    originals=[a.copy() for a in (x,y,p)]
    base=Sampler([(x,y)],435,16);mined=ErrorSampler([(x,y)],[p],435,16)
    pos=mined.positive[0][0];neg=mined.negative[0][0]
    assert ((pos==[10,10]).all(1)).mean()>=.5
    assert ((neg==[1,1]).all(1)).mean()>=.5
    assert y[0,pos[:,0],pos[:,1]].all()
    assert not y[0,neg[:,0],neg[:,1]].any()
    assert len(pos)==2*len(base.positive[0][0])
    for a,b in zip((x,y,p),originals):np.testing.assert_array_equal(a,b)


def test_perfect_predictions_leave_candidate_pools_unchanged_and_bad_cache_rejected():
    x=np.random.default_rng(8).random((2,24,24),dtype=np.float32);y=x>.8
    base=Sampler([(x,y)],435,16);mined=ErrorSampler([(x,y)],[y.astype(float)],435,16)
    for _ in range(20):
        a,b,c=base.sample(1);d,e,f=mined.sample(1)
        assert c==f;np.testing.assert_array_equal(a,d);np.testing.assert_array_equal(b,e)
    with pytest.raises(ValueError):ErrorSampler([(x,y)],[np.zeros((1,24,24))],435,16)
