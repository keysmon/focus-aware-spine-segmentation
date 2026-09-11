import numpy as np
from segmentation.focus.data import Sampler
from experiments.unaugmented_sampling import UnaugmentedSampler


def test_raw_crop_and_preserved_schedule():
    x=np.random.default_rng(9).random((4,19,23),dtype=np.float32);y=x>.6
    base=Sampler([(x,y)],435,16);candidate=UnaugmentedSampler([(x,y)],435,16)
    for _ in range(30):
        _,_,bm=base.sample(1);a,b,m=candidate.sample(1)
        assert bm==m
        z,t,l=m['z'],m['top'],m['left']
        np.testing.assert_array_equal(a[0],x[z,t:t+16,l:l+16])
        np.testing.assert_array_equal(b,y[z,t:t+16,l:l+16])
    assert candidate.schedule_hash()==base.schedule_hash()
