import numpy as np
import pytest
from segmentation.focus.data import normalize_stack, context, Sampler


def test_context_boundary_and_center():
    x = np.arange(3, dtype=np.float32)[:, None, None]
    assert context(x, 0, 5)[:, 0, 0].tolist() == [0, 0, 0, 1, 2]
    assert context(x, 2, 5)[:, 0, 0].tolist() == [0, 1, 2, 2, 2]
    np.testing.assert_array_equal(context(x, 1, 1)[0], x[1])
    with pytest.raises(ValueError): context(x, 0, 2)


def test_stack_normalization_preserves_relative_intensity():
    x = np.stack([np.full((16,16), v) for v in [10, 20, 30]])
    y, bounds = normalize_stack(x)
    assert np.allclose(y[:,0,0], [0,0.5,1])
    assert bounds == [10.0,30.0]
    assert not normalize_stack(np.ones((3,8,8)))[0].any()


def test_sampler_schedule_and_center_masks_match():
    y = np.zeros((4,32,32), bool); y[:,8:20,11:21] = True
    x = y.astype(np.float32)
    a, b = Sampler([(x,y)],435,32), Sampler([(x,y)],435,32)
    for _ in range(30):
        single, m1, meta1 = a.sample(1)
        multi, m2, meta2 = b.sample(5)
        assert meta1 == meta2
        np.testing.assert_array_equal(single[0], multi[2])
        np.testing.assert_array_equal(m1,m2)
        np.testing.assert_array_equal(multi[2]>0.5,m2)
    assert a.schedule_hash() == b.schedule_hash()
    assert m1.any(), 'Negative-centered crops must retain real foreground'


def test_sampler_no_cross_stack_context():
    pairs=[(np.full((3,16,16),v,np.float32),np.zeros((3,16,16),bool)) for v in [0.,1.]]
    sampler=Sampler(pairs,435,16)
    for _ in range(20):
        x,y,meta=sampler.sample(5)
        assert np.max(np.ptp(x,axis=0)) == 0
        assert not y.any()
    assert sampler.counts['fallback'] > 0


def test_center_supervision_with_different_masks_per_frame():
    y=np.zeros((3,32,32),bool)
    y[0,3:9,3:9]=True; y[1,12:18,12:18]=True; y[2,23:29,23:29]=True
    x=y.astype(np.float32)
    sampler=Sampler([(x,y)],435,32)
    centers=set()
    for _ in range(40):
        inputs,target,meta=sampler.sample(5)
        centers.add(meta['z'])
        np.testing.assert_array_equal(inputs[2]>.5,target)
    assert centers=={0,1,2}
