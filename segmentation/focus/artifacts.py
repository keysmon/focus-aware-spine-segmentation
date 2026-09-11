"""Frozen experiment contracts and hash checks."""
import json
from pathlib import Path
from segmentation.data import digest
from segmentation.report import write_json
from segmentation.splits import FOLDS


def freeze_manifest(paths,destination):
    write_json(destination,{str(Path(p).resolve()):digest(p) for p in paths})


def verify_manifest(path):
    values=json.loads(Path(path).read_text())
    for name,expected in values.items():
        if not Path(name).is_file() or digest(name)!=expected:
            raise ValueError(f'Frozen artifact changed: {name}')


def validate_config(config):
    expected=[dict(f,train=list(f['train'])) for f in FOLDS]
    if config.get('folds')!=expected: raise ValueError('Invalid fold membership')
    if config.get('methods')!={'R1':1,'R2':5}: raise ValueError('Expected paired one/five-slice methods')
    train=config['training']
    for name in ('epochs','batches_per_epoch','batch_size','training_seconds','patience','patch_size','stride'):
        if train[name]<=0: raise ValueError(f'Expected positive {name}')
    if train['stride']>train['patch_size']: raise ValueError('Stride exceeds patch size')
    if not train['thresholds'] or any(not 0<t<1 for t in train['thresholds']):
        raise ValueError('Invalid thresholds')
