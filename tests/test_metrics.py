import numpy as np
import pytest
from segmentation.metrics import evaluate


def test_confusion_metrics():
    y = np.array([1, 1, 0, 0], bool).reshape(1, 2, 2)
    p = np.array([1, 0, 1, 0], bool).reshape(1, 2, 2)
    r = evaluate(y, p)
    assert r['dice'] == r['precision'] == r['recall'] == 0.5
    assert r['fp'] == r['fn'] == 1


def test_empty_masks():
    x = np.zeros((1, 8, 8), bool)
    r = evaluate(x, x)
    assert r['dice'] == 1 and r['precision'] is None and r['recall'] is None
    assert r['boundary_mean_px'] is None and r['both_empty_slices'] == 1
    y = x.copy(); y[0, 3, 3] = True
    r = evaluate(x, y)
    assert r['dice'] == 0 and r['one_empty_slices'] == 1


def test_displaced_boundaries():
    x = np.zeros((1, 8, 8), bool); x[0, 3, 3] = True
    y = np.zeros_like(x); y[0, 3, 4] = True
    assert evaluate(x, y)['boundary_mean_px'] == 1
    assert evaluate(x, x)['boundary_mean_px'] == 0


def test_shape_rejected():
    with pytest.raises(ValueError):
        evaluate(np.zeros((1, 2, 2), bool), np.zeros((2, 2, 2), bool))
