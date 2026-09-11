"""Stack overlap and slice-wise symmetric boundary distances in pixels."""
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def evaluate(reference, prediction):
    y, p = np.asarray(reference, bool), np.asarray(prediction, bool)
    if y.shape != p.shape or y.ndim != 3 or not y.size:
        raise ValueError('Expected matching nonempty stack shapes')
    tp, fp, fn, tn = (int(a.sum()) for a in (y & p, ~y & p, y & ~p, ~y & ~p))
    means, p95s = [], []
    both_empty = one_empty = 0
    for a, b in zip(y, p):
        if not a.any() or not b.any():
            both_empty += int(not a.any() and not b.any())
            one_empty += int(a.any() != b.any())
            continue
        ab = a & ~binary_erosion(a, border_value=0)
        bb = b & ~binary_erosion(b, border_value=0)
        distances = np.concatenate((distance_transform_edt(~bb)[ab], distance_transform_edt(~ab)[bb]))
        means.append(float(distances.mean()))
        p95s.append(float(np.percentile(distances, 95)))
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
        dice=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / (tp + fn) if tp + fn else None,
        precision_available=int(tp + fp > 0), recall_available=int(tp + fn > 0),
        accuracy=(tp + tn) / y.size,
        boundary_mean_px=float(np.mean(means)) if means else None,
        boundary_p95_px=float(np.mean(p95s)) if p95s else None,
        eligible_boundary_slices=len(means), both_empty_slices=both_empty, one_empty_slices=one_empty)
