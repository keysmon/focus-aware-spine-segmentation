import numpy as np
import pytest
from experiments.focal_calibration import relative_confidence


def test_zero_and_weak_sweeps_do_not_become_confident_foreground():
    assert np.count_nonzero(relative_confidence(np.zeros((3,4,5))))==0
    x=np.full((3,4,5),.01,np.float32)
    assert relative_confidence(x).max()<.04


def test_confidence_retains_shape_order_and_bounds():
    x=np.array([.05,.2,.1],np.float32)[:,None,None]
    original=x.copy()
    p=relative_confidence(x)
    assert p.shape==x.shape and p.dtype==np.float32
    assert 0<=p.min()<=p.max()<=1
    assert p.argmax()==x.argmax() and np.isclose(p[1,0,0],.5)
    np.testing.assert_array_equal(x,original)


def test_invalid_probabilities_rejected():
    for x in (np.ones((2,3)),np.array([[[np.nan]]]),np.array([[[1.1]]])):
        with pytest.raises(ValueError):relative_confidence(x)
