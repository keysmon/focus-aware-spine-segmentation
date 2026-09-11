import json
import numpy as np
import pytest
from segmentation.report import create_run, verify_inputs, freeze_config, write_panels


def test_duplicate_run_is_rejected(tmp_path):
    output = tmp_path / 'same-run'; output.mkdir()
    with pytest.raises(FileExistsError): create_run(output)


def test_changed_input_rejected(tmp_path):
    (tmp_path / 'project.py').write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        verify_inputs(tmp_path, {'project.py': 'wrong'})


def test_frozen_config_rejects_overlap(tmp_path):
    config = {'folds': [{'test':'2', 'validation':'3', 'train':['2','7']}]}
    with pytest.raises(ValueError, match='fold'):
        freeze_config(config, tmp_path / 'config.json')


def test_panels_write_file(tmp_path):
    x = np.zeros((2, 32, 32), np.uint8)
    y = x.astype(bool); y[:, 8:12, 8:12] = True
    write_panels(x, y, y, tmp_path / 'panels.png')
    assert (tmp_path / 'panels.png').stat().st_size > 100


def test_frozen_config_cannot_be_replaced(tmp_path):
    from segmentation.splits import FOLDS
    config = {'folds': [dict(f, train=list(f['train'])) for f in FOLDS]}
    destination = tmp_path/'config.json'
    freeze_config(config, destination)
    with pytest.raises(FileExistsError): freeze_config(config, destination)


def test_saved_prediction_preserves_tiff_shape(tmp_path):
    from segmentation.__main__ import save_prediction
    import tifffile
    x = np.zeros((2, 32, 32), np.uint8)
    y = x.astype(bool); y[:, 8:12, 8:12] = True
    p = y.astype(np.uint8)*255
    metrics = save_prediction(tmp_path, 'tiny', x, y, p)
    np.testing.assert_array_equal(tifffile.imread(tmp_path/'tiny_prediction.tiff'), p)
    assert metrics['dice'] == 1
