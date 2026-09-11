import numpy as np
from experiments.geometric_sampling import warp_pair,GeometricSampler


def test_identity_warp_is_exact_crop():
    image=np.arange(30*40,dtype=np.float32).reshape(30,40)
    mask=image%7==0
    x,y=warp_pair(image,mask,3,5,16,0,1)
    np.testing.assert_array_equal(x,image[3:19,5:21])
    np.testing.assert_array_equal(y,mask[3:19,5:21])


def test_rotated_scaled_landmark_stays_aligned_and_mask_binary():
    mask=np.zeros((32,32),bool)
    mask[9:16,12:19]=True
    x,y=warp_pair(mask.astype(np.float32),mask,0,0,32,33,1.2)
    assert y.dtype==bool and y.any()
    assert x[y].mean()>.9
    assert x[~y].mean()<.02


def test_augmented_sampler_reproducible_without_mutating_sources():
    x=np.random.default_rng(1).random((3,32,32)).astype(np.float32)
    y=x>.8
    original=x.copy()
    a,b=GeometricSampler([(x,y)],435,16),GeometricSampler([(x,y)],435,16)
    for _ in range(4):
        ax,ay,am=a.sample()
        bx,by,bm=b.sample()
        np.testing.assert_array_equal(ax,bx)
        np.testing.assert_array_equal(ay,by)
        assert am==bm and a.schedule_hash()==b.schedule_hash()
    np.testing.assert_array_equal(x,original)
