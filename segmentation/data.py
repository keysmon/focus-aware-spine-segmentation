"""Strict paired TIFF loading; never changes source assets."""
import hashlib
from pathlib import Path
import numpy as np
import tifffile

STACK_IDS = ('2', '3', '6', '7')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_stack(root: Path, stack_id: str):
    root = Path(root)
    x = tifffile.imread(root / f'{stack_id}_image.tiff')
    y = tifffile.imread(root / f'{stack_id}_mask.tiff')
    if x.ndim != 3 or x.shape != y.shape or not x.size:
        raise ValueError(f'{stack_id}: incompatible/non-3D shape {x.shape}, {y.shape}')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f'{stack_id}: image and mask must be finite')
    values = set(np.unique(y).tolist())
    if not (values <= {0, 1} or values <= {0, 255}):
        raise ValueError(f'{stack_id}: unexpected mask values {values}')
    return x, y != 0


def audit(root: Path):
    root = Path(root)
    result = {'provenance': 'Acquisition independence unverified', 'stacks': {}}
    for sid in STACK_IDS:
        x, y = load_stack(root, sid)
        result['stacks'][sid] = dict(shape=list(x.shape), dtype=str(x.dtype),
            minimum=float(x.min()), maximum=float(x.max()), foreground_fraction=float(y.mean()),
            mask_values=np.unique(tifffile.imread(root / f'{sid}_mask.tiff')).tolist(),
            hashes={str(p.name): digest(p) for p in [root / f'{sid}_image.tiff', root / f'{sid}_mask.tiff']})
    result['original_hashes'] = {p.name: digest(p) for p in sorted(root.iterdir())
        if p.suffix in ('.tiff', '.ipynb') or p.name in ('project.py', 'data_import.py')}
    return result
