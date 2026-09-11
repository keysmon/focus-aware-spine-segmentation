import numpy as np
import pytest
from experiments.full_range import normalize_stack


def test_preserves_distinct_dark_values_and_constant_stack():
    x=np.arange(256,dtype=np.uint8).reshape(1,16,16)
    y,bounds=normalize_stack(x)
    assert bounds==[0,255] and y.dtype==np.float32
    assert np.all(np.diff(y.ravel())>0)
    np.testing.assert_allclose(y*255,x,atol=1e-5)
    z,b=normalize_stack(np.ones((2,3,4)))
    assert not z.any() and b==[1,1]


def test_rejects_invalid_arrays():
    for x in (np.zeros((2,3)),np.zeros((0,2,3)),np.full((1,2,3),np.nan)):
        with pytest.raises(ValueError):normalize_stack(x)
