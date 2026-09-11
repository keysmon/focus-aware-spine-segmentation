import numpy as np
import pytest
from experiments.local_contrast import enhance_stack, ContrastSampler
from segmentation.focus.data import Sampler


def test_local_contrast_preserves_geometry_and_does_not_mix_planes():
    x=np.random.default_rng(8).uniform(.2,.7,(2,70,90)).astype(np.float32)
    original=x.copy(); result=enhance_stack(x)
    assert result.shape==x.shape and result.dtype==np.float32
    assert np.isfinite(result).all() and result.min()>=0 and result.max()<=1
    np.testing.assert_array_equal(x,original)
    np.testing.assert_array_equal(result[:1],enhance_stack(x[:1]))
    assert not np.array_equal(x,result)
    with pytest.raises(ValueError): enhance_stack(np.array([[[np.nan]]]))
    with pytest.raises(ValueError): enhance_stack(np.ones((2,3)))
    with pytest.raises(ValueError): enhance_stack(np.full((1,3,3),2.))


def test_contrast_sampling_keeps_original_coordinates_and_mask_augmentation():
    rng=np.random.default_rng(9);x=rng.random((3,45,50),dtype=np.float32)
    y=x>.8;original=x.copy();base=Sampler([(x,y)],435,32)
    modified=ContrastSampler([(x,y)],435,32)
    for _ in range(30):
        _,a,meta=base.sample(1);image,b,other=modified.sample(1)
        assert meta==other
        np.testing.assert_array_equal(a,b)
        assert image.shape==(1,32,32)
    assert base.schedule_hash()==modified.schedule_hash()
    np.testing.assert_array_equal(x,original)
