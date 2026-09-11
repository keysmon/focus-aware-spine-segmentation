import cv2
import numpy as np
from segmentation.focus.data import Sampler
from experiments.focus_rotation import rotate_pair, RotationSampler


def test_rotation_preserves_mask_image_alignment_and_binary_labels():
    y=np.zeros((160,160),bool);y[35:125,60:100]=True;x=y.astype(np.float32)
    image,mask=rotate_pair(x,y,16,16,128,45)
    assert image.dtype==np.float32 and mask.dtype==bool
    assert image.shape==mask.shape==(128,128) and image.min()>=0 and image.max()<=1
    intersection=((image>.5)&mask).sum();union=((image>.5)|mask).sum()
    assert intersection/union>.99


def test_rotation_sampler_retains_sampling_locations_and_photometry_draws():
    rng=np.random.default_rng(7);x=rng.random((3,145,150),dtype=np.float32);y=x>.8
    base=Sampler([(x,y)],435);aug=RotationSampler([(x,y)],435)
    extra=[]
    for _ in range(20):
        a,b,m=base.sample(1);c,d,n=aug.sample(1)
        for key,value in m.items():assert n[key]==value
        extra.append(n.get('extra_angle',0))
        assert c.shape==a.shape and d.dtype==bool and np.isfinite(c).all()
        if not n.get('extra_angle',0):
            np.testing.assert_array_equal(c,a);np.testing.assert_array_equal(d,b)
    assert set(extra)=={0,45}
