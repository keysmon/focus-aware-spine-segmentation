import numpy as np
import pytest
from segmentation.focus.data import Sampler
from experiments.gamma_sampling import gamma_transform,GammaSampler


def test_gamma_is_pointwise_monotone_bounded_and_preserves_input():
    x=np.linspace(0,1,100,dtype=np.float32).reshape(1,10,10);before=x.copy()
    for gamma in (.7,1.,1.5):
        y=gamma_transform(x,gamma)
        assert y.shape==x.shape and y.dtype==np.float32
        assert (np.diff(y.ravel())>=0).all() and y.min()==0 and y.max()==1
    np.testing.assert_array_equal(gamma_transform(x,1),x);np.testing.assert_array_equal(before,x)
    with pytest.raises(ValueError):gamma_transform(x,0)


def test_gamma_sampler_keeps_original_masks_and_sampling_draws():
    x=np.random.default_rng(9).random((3,40,45),dtype=np.float32);y=x>.8
    base=Sampler([(x,y)],435,32);aug=GammaSampler([(x,y)],435,32);gammas=[]
    for _ in range(30):
        a,b,m=base.sample(1);c,d,n=aug.sample(1)
        for k,v in m.items():assert n[k]==v
        np.testing.assert_array_equal(b,d)
        gamma=n.get('gamma',1.);gammas.append(gamma)
        np.testing.assert_allclose(c,a**gamma,rtol=1e-6,atol=1e-7)
    assert min(gammas)<1<max(gammas) and 1 in gammas
