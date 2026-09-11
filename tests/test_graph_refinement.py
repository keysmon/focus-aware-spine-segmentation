import itertools
import numpy as np
import pytest
from experiments.graph_refinement import refine_plane


def test_graph_cut_matches_exhaustive_binary_energy():
    image=np.array([[.2,.25,.8],[.15,.3,.9]])
    p=np.array([[.9,.4,.15],[.75,.35,.1]])
    threshold=.3;strength=2.
    pairs=[(a,b) for a in range(6) for b in range(a+1,6) if (b==a+1 and a//3==b//3) or b==a+3]
    diffs=np.array([(image.flat[a]-image.flat[b])**2 for a,b in pairs])
    weights=strength*np.exp(-diffs/(2*diffs.mean()+1e-12))
    score=np.log(p/(1-p))-np.log(threshold/(1-threshold))
    def energy(y):
        return float((-score*y).sum()+sum(w*(y.flat[a]!=y.flat[b]) for (a,b),w in zip(pairs,weights)))
    result=refine_plane(image,p,threshold,strength)
    optimum=min(energy(np.array(y).reshape(2,3)) for y in itertools.product([0,1],repeat=6))
    assert energy(result)==pytest.approx(optimum,abs=.001)


def test_graph_cut_zero_strength_and_validation():
    p=np.array([[0.,.3,1.],[.2,.6,.7]])
    np.testing.assert_array_equal(refine_plane(p,p,.3,0),p>=.3)
    for value in (0.,1.):
        x=np.full((3,4),value)
        np.testing.assert_array_equal(refine_plane(x,x,.3,2),x.astype(bool))
    with pytest.raises(ValueError):refine_plane(p,p[:1],.3,2)
    with pytest.raises(ValueError):refine_plane(p,p,.3,-1)
    with pytest.raises(ValueError):refine_plane(p,p,1,2)
