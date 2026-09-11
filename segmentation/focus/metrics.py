"""Focus-specific failures plus comparable boundary denominators."""
import numpy as np
from segmentation.metrics import evaluate


def selection_score(reference,prediction):
    y,p=np.asarray(reference,bool),np.asarray(prediction,bool)
    if y.shape!=p.shape or y.ndim!=3 or not y.size:
        raise ValueError('Expected matching nonempty ZHW arrays')
    tp=int((y&p).sum()); fp=int((~y&p).sum()); fn=int((y&~p).sum())
    dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.
    empty=~y.any(axis=(1,2))
    empty_fp=float(p[empty].mean()) if empty.any() else 0.
    return dice,-empty_fp


def diagnostics(reference,prediction):
    y,p=np.asarray(reference,bool),np.asarray(prediction,bool)
    result=evaluate(y,p)
    empty=~y.any(axis=(1,2)); active=p.any(axis=(1,2))
    result.update(empty_slices=int(empty.sum()),nonempty_slices=int((~empty).sum()),
        empty_activated_slices=int((empty&active).sum()),
        empty_activation_fraction=float(active[empty].mean()) if empty.any() else None,
        empty_fp_pixel_fraction=float(p[empty].mean()) if empty.any() else None)
    rows=[]
    for z,(a,b) in enumerate(zip(y,p)):
        r=evaluate(a[None],b[None]); r.update(index=z,reference_pixels=int(a.sum()),prediction_pixels=int(b.sum()))
        rows.append(r)
    result['slices']=rows
    foreground=[r for r in rows if r['reference_pixels']]
    result['nonempty_mean_recall']=float(np.mean([r['recall'] for r in foreground])) if foreground else None
    result['nonempty_mean_dice']=float(np.mean([r['dice'] for r in foreground])) if foreground else None
    return result


def comparison_metrics(reference,predictions):
    common=np.asarray(reference,bool).any(axis=(1,2))
    for p in predictions.values(): common &= np.asarray(p,bool).any(axis=(1,2))
    result={}
    for name,p in predictions.items():
        r=diagnostics(reference,p)
        shared=evaluate(reference[common],p[common]) if common.any() else None
        r['shared_boundary_slices']=np.flatnonzero(common).tolist()
        r['shared_boundary_mean_px']=shared['boundary_mean_px'] if shared else None
        r['shared_boundary_p95_px']=shared['boundary_p95_px'] if shared else None
        result[name]=r
    return result
