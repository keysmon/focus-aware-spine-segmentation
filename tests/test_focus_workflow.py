import pytest
from pathlib import Path
from segmentation.focus.artifacts import freeze_manifest, verify_manifest, validate_config


def test_manifest_detects_changes(tmp_path):
    p=tmp_path/'data'; p.write_text('original')
    manifest=tmp_path/'manifest.json'
    freeze_manifest([p],manifest)
    verify_manifest(manifest)
    p.write_text('changed')
    with pytest.raises(ValueError,match='changed'): verify_manifest(manifest)


def test_manifest_refuses_overwrite(tmp_path):
    manifest=tmp_path/'manifest.json'
    freeze_manifest([],manifest)
    with pytest.raises(FileExistsError): freeze_manifest([],manifest)


def test_invalid_fold_rejected():
    with pytest.raises(ValueError,match='fold'): validate_config({'folds':[]})
