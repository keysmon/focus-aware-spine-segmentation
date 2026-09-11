import numpy as np
from segmentation.focus.metrics import diagnostics, comparison_metrics, selection_score


def test_empty_groups_and_rates():
    y=np.zeros((2,4,4),bool); p=y.copy(); p[0,0,0]=True
    r=diagnostics(y,p)
    assert r['empty_activation_fraction']==0.5
    assert r['empty_fp_pixel_fraction']==1/32
    assert r['nonempty_mean_recall'] is None
    y[1,1,1]=True
    r=diagnostics(y,p)
    assert r['nonempty_mean_recall']==0
    assert len(r['slices'])==2
    assert r['empty_fp_pixel_fraction']==1/16


def test_shared_boundary_subset():
    y=np.zeros((3,5,5),bool); y[:,2,2]=True
    a=y.copy(); b=y.copy(); b[0]=False
    scores=comparison_metrics(y,{'a':a,'b':b})
    for r in scores.values():
        assert r['shared_boundary_slices']==[1,2]
        assert r['shared_boundary_mean_px']==0


def test_selection_tie_prefers_fewer_empty_false_positives():
    y=np.zeros((2,4,4),bool)
    assert selection_score(y,y) > selection_score(y,~y)


def test_equal_dice_tie_uses_empty_slice_pixels():
    y=np.zeros((2,4,4),bool); y[0,0,0]=True
    a=y.copy(); a[0,0,1]=True
    b=y.copy(); b[1,0,1]=True
    assert selection_score(y,a)[0]==selection_score(y,b)[0]
    assert selection_score(y,a)>selection_score(y,b)
