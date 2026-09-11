import numpy as np
from experiments.dihedral_prediction import dihedral_predictor


def test_inverse_coordinates_and_all_eight_views():
    x=np.random.default_rng(4).random((2,1,9,13),dtype=np.float32);seen=[]
    def predict(a):
        assert a.flags.c_contiguous
        seen.append(a.copy());return a.copy()
    result=dihedral_predictor(predict)(x)
    np.testing.assert_allclose(result,x,atol=2e-7)
    assert len(seen)==8
    assert len({a.tobytes() for a in seen})==8
