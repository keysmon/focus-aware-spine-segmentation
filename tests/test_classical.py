import numpy as np
from project import generate_segmentation_map
from segmentation.classical import baseline, classical, CANDIDATES


def test_adapter_preserves_existing_algorithm():
    x = np.random.default_rng(435).integers(0, 256, (3, 48, 48), dtype=np.uint8)
    np.testing.assert_array_equal(baseline(x), generate_segmentation_map(x))
    np.testing.assert_array_equal(classical(x, CANDIDATES[0]), baseline(x))


def test_candidates_binary_and_constant_empty():
    for config in CANDIDATES[1:]:
        x = np.zeros((2, 48, 48), np.uint8)
        y = classical(x, config)
        assert y.shape == x.shape and y.dtype == np.uint8 and not y.any()
        x[:, 16:32, 16:32] = 100
        assert set(np.unique(classical(x, config))) <= {0, 255}
