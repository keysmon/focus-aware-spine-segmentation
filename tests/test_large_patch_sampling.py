import numpy as np
from experiments.large_patch_sampling import Sampler


def test_padding_preserves_positive_image_mask_correspondence():
    y=np.zeros((2,20,24),bool)
    y[:,5:18,15:24]=True
    sampler=Sampler([(y.astype(np.float32),y)],seed=435,size=32)
    for _ in range(20):
        x,mask,_=sampler.sample(1)
        assert x.shape==(1,32,32) and mask.shape==(32,32)
        assert np.array_equal(x[0]>.5,mask)
        assert mask.dtype==bool and x.dtype==np.float32


def test_padding_does_not_mutate_original_pairs():
    x=np.arange(120,dtype=np.float32).reshape(2,6,10)/120
    y=x>.5
    old_x,old_y=x.copy(),y.copy()
    sampler=Sampler([(x,y)],seed=3,size=16)
    sampler.sample(1)
    assert np.array_equal(x,old_x) and np.array_equal(y,old_y)
