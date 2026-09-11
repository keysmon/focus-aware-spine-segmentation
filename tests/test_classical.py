import numpy as np
import hashlib
from segmentation.classical import baseline, classical, CANDIDATES


# Regression fixture from the original baseline on the fixed random stack.
BASELINE_SHA256 = 'b2940d4ce42c8704fda34fd634c531b4803470dd2367675476665267275c7d5b'

def test_adapter_preserves_existing_algorithm():
    x = np.random.default_rng(435).integers(0, 256, (3, 48, 48), dtype=np.uint8)
    assert hashlib.sha256(baseline(x).tobytes()).hexdigest() == BASELINE_SHA256
    np.testing.assert_array_equal(classical(x, CANDIDATES[0]), baseline(x))


def test_candidates_binary_and_constant_empty():
    for config in CANDIDATES[1:]:
        x = np.zeros((2, 48, 48), np.uint8)
        y = classical(x, config)
        assert y.shape == x.shape and y.dtype == np.uint8 and not y.any()
        x[:, 16:32, 16:32] = 100
        assert set(np.unique(classical(x, config))) <= {0, 255}
