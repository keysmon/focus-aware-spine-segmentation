"""Run with python -m segmentation; phases never overwrite completed work."""
import argparse
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ece435-mpl')
import numpy as np
import tifffile
from segmentation.data import audit, load_stack, digest, STACK_IDS
from segmentation.classical import baseline, classical, select_classical, CANDIDATES, normalize
from segmentation.metrics import evaluate
from segmentation.patches import predict_tiled
from segmentation.report import create_run, write_json, verify_inputs, freeze_config, write_panels, summarize
from segmentation.splits import FOLDS
from segmentation.train import smoke, train_fold, load_predictor, device_name


def save_prediction(directory, sid, images, reference, prediction):
    if prediction.shape != images.shape or prediction.dtype != np.uint8 or not set(np.unique(prediction)) <= {0,255}:
        raise ValueError('Invalid prediction shape, dtype or values')
    tifffile.imwrite(directory/f'{sid}_prediction.tiff', prediction, photometric='minisblack')
    write_panels(images, reference, prediction != 0, directory/f'{sid}_panels.png')
    return evaluate(reference, prediction != 0)


def source_hashes():
    return {str(p): digest(p) for p in sorted(Path('segmentation').glob('*.py'))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['audit','fit','evaluate','report'])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('.'))
    parser.add_argument('--config', type=Path, default=Path('configs/comparison.json'))
    args = parser.parse_args(); run = args.run
    if args.phase == 'audit':
        create_run(run)
        data = audit(args.data)
        data['data_root'] = str(args.data.resolve())
        data['source_hashes'] = source_hashes()
        data['python'] = platform.python_version()
        data['dependencies'] = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
        write_json(run/'audit.json', data)
        directory = run/'baseline'; directory.mkdir()
        metrics = {}; start = time.monotonic()
        for sid in STACK_IDS:
            x, y = load_stack(args.data, sid)
            metrics[sid] = save_prediction(directory, sid, x, y, baseline(x))
            print(sid, metrics[sid], flush=True)
        write_json(run/'baseline-metrics.json', metrics)
        write_json(run/'audit-complete.json', {'seconds':time.monotonic()-start})
        return
    data = json.loads((run/'audit.json').read_text()); root = Path(data['data_root'])
    verify_inputs(root, data['original_hashes'])
    if args.phase == 'fit':
        if not (run/'audit-complete.json').exists(): raise ValueError('Audit incomplete')
        config = json.loads(args.config.read_text())
        freeze_config(config, run/'config.json')
        write_json(run/'frozen-source.json', source_hashes())
        write_json(run/'config-hash.json', {'sha256':digest(run/'config.json')})
        smoke(run/'smoke.json')
        start = time.monotonic(); deadline = start + config['run_seconds']
        selections = []
        for fold in FOLDS:
            if time.monotonic() >= deadline: raise TimeoutError('Overall experiment time limit; partial artifacts preserved')
            sid = fold['test']; print('Fitting test fold', sid, flush=True)
            validation = load_stack(root, fold['validation'])
            selected_classical = select_classical(*validation, CANDIDATES)
            neural = train_fold([load_stack(root, s) for s in fold['train']], validation,
                                config['training'], run/f'fold-{sid}', deadline)
            selection = dict(test=sid, validation=fold['validation'], train=list(fold['train']),
                             classical=selected_classical, neural=neural)
            write_json(run/f'fold-{sid}'/'frozen-selection.json', selection)
            selections.append(selection)
        write_json(run/'fit-complete.json', dict(folds=selections, elapsed_seconds=time.monotonic()-start))
    elif args.phase == 'evaluate':
        if not (run/'fit-complete.json').exists(): raise ValueError('All selections must be frozen before evaluation')
        verify_inputs(Path('.'), json.loads((run/'frozen-source.json').read_text()))
        if digest(run/'config.json') != json.loads((run/'config-hash.json').read_text())['sha256']:
            raise ValueError('Config changed after fitting')
        directory = run/'held-out'; directory.mkdir(exist_ok=False)
        config = json.loads((run/'config.json').read_text())['training']
        metrics = {'baseline':json.loads((run/'baseline-metrics.json').read_text()), 'classical':{}, 'unet':{}}
        for method in ('classical','unet'): (directory/method).mkdir()
        for sid in STACK_IDS:
            fold_dir = run/f'fold-{sid}'
            selection = json.loads((fold_dir/'frozen-selection.json').read_text())
            x, y = load_stack(root, sid)
            p = classical(x, selection['classical']['selected'])
            metrics['classical'][sid] = save_prediction(directory/'classical', sid, x, y, p)
            predict = load_predictor(fold_dir/'best.pt', device_name())
            prob = predict_tiled(normalize(x), predict, config['patch_size'], config['stride'])
            p = (prob >= selection['neural']['selected']['threshold']).astype(np.uint8)*255
            metrics['unet'][sid] = save_prediction(directory/'unet', sid, x, y, p)
            print('Evaluated', sid, metrics['unet'][sid], flush=True)
        write_json(run/'metrics.json', metrics)
    else:
        print(json.dumps(summarize(run), indent=2))


if __name__ == '__main__':
    main()
