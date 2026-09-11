import numpy as np
import pytest
import tifffile
from segmentation.data import load_stack, audit
from segmentation.splits import FOLDS


def write_pair(root, x, y):
    tifffile.imwrite(root / '2_image.tiff', x, photometric='minisblack')
    tifffile.imwrite(root / '2_mask.tiff', y, photometric='minisblack')


def test_rejects_misaligned_stack(tmp_path):
    write_pair(tmp_path, np.zeros((3, 8, 8), np.uint8), np.zeros((2, 8, 8), np.uint8))
    with pytest.raises(ValueError, match='shape'):
        load_stack(tmp_path, '2')


@pytest.mark.parametrize('value', [1, 255])
def test_mask_encodings(tmp_path, value):
    x = np.ones((2, 8, 8), np.uint8)
    write_pair(tmp_path, x, x * value)
    image, mask = load_stack(tmp_path, '2')
    np.testing.assert_array_equal(image, x)
    assert mask.dtype == bool and mask.all()


def test_rejects_unknown_mask_values(tmp_path):
    write_pair(tmp_path, np.zeros((2, 8, 8), np.uint8), np.full((2, 8, 8), 7, np.uint8))
    with pytest.raises(ValueError, match='mask'):
        load_stack(tmp_path, '2')


def test_rejects_nonfinite_images(tmp_path):
    write_pair(tmp_path, np.full((2, 8, 8), np.nan), np.zeros((2, 8, 8), np.uint8))
    with pytest.raises(ValueError, match='finite'):
        load_stack(tmp_path, '2')


def test_folds_disjoint_and_exhaustive():
    assert [f['test'] for f in FOLDS] == ['2', '3', '6', '7']
    for f in FOLDS:
        assert len(set((*f['train'], f['validation'], f['test']))) == 4
        assert set((*f['train'], f['validation'], f['test'])) == {'2', '3', '6', '7'}
