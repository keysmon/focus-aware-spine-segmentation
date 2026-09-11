"""Sequential audit, fit, frozen evaluation and reporting for focal context."""
import argparse
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/ece435-mpl')
import numpy as np
import tifffile
import torch
from segmentation.data import load_stack, audit, STACK_IDS
from segmentation.report import create_run, write_json, verify_inputs, write_panels
from segmentation.splits import FOLDS
from segmentation.train import device_name
from segmentation.focus.data import normalize_stack
from segmentation.focus.artifacts import freeze_manifest, verify_manifest, validate_config
from segmentation.focus.train import train_fold, smoke
from segmentation.focus.model import predict_stack, load_predictor
from segmentation.focus.metrics import comparison_metrics
from segmentation.focus.visuals import location, axial_panel, overview
from segmentation.focus.report import make_report


def versions():
    return {d.metadata['Name']:d.version for d in importlib.metadata.distributions()}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['audit','fit','evaluate','report'])
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--data',type=Path,default=Path('data'))
    parser.add_argument('--reference',type=Path,default=Path('runs/comparison-001'))
    parser.add_argument('--config',type=Path,default=Path('configs/focus-comparison.json'))
    args=parser.parse_args(); run=args.run
    torch.set_num_threads(2)
    if args.phase=='audit':
        config=json.loads(args.config.read_text()); validate_config(config)
        prior=json.loads((args.reference/'audit.json').read_text())
        verify_inputs(args.data,prior['original_hashes'])
        verify_inputs(Path('.'),json.loads((args.reference/'frozen-source.json').read_text()))
        create_run(run)
        write_json(run/'config.json',config)
        data=audit(args.data)
        data.update(data_root=str(args.data.resolve()),reference=str(args.reference.resolve()),
                    dependencies=versions(),python=platform.python_version(),bounds={},locations={})
        (run/'audit-panels').mkdir()
        for sid in STACK_IDS:
            x,y=load_stack(args.data,sid)
            data['bounds'][sid]=normalize_stack(x)[1]; data['locations'][sid]=location(y)
            axial_panel(x,y,{},data['locations'][sid],run/'audit-panels'/f'{sid}.png')
        write_json(run/'audit.json',data)
        files=[args.data/name for name in data['original_hashes']]
        freeze_manifest(files,run/'inputs-manifest.json')
        freeze_manifest([p for p in args.reference.rglob('*') if p.is_file()],run/'reference-manifest.json')
        freeze_manifest([*Path('segmentation').rglob('*.py'),Path('requirements-lock.txt'),run/'config.json'],run/'source-manifest.json')
        print('Audit complete',run)
        return
    data=json.loads((run/'audit.json').read_text()); root=Path(data['data_root']); reference=Path(data['reference'])
    for name in ('inputs','reference','source'): verify_manifest(run/f'{name}-manifest.json')
    if data['dependencies']!=versions() or data['python']!=platform.python_version():
        raise ValueError('Environment changed since audit')
    config=json.loads((run/'config.json').read_text()); validate_config(config)
    if args.phase=='fit':
        (run/'fitting').mkdir(exist_ok=False)
        smoke(run/'smoke.json')
        start=time.monotonic(); deadline=start+config['run_seconds']; fits=[]
        for fold in FOLDS:
            pairs=[load_stack(root,sid) for sid in fold['train']]
            validation=load_stack(root,fold['validation'])
            results=[]
            for method,channels in config['methods'].items():
                if time.monotonic()>=deadline: raise TimeoutError('Overall fit budget reached; partial files preserved')
                print(f'FIT {method} test={fold["test"]}',flush=True)
                path=run/'fitting'/f'{method}-{fold["test"]}'
                result=train_fold(pairs,validation,config['training'],channels,path,deadline)
                record=dict(method=method,test=fold['test'],validation=fold['validation'],train=list(fold['train']),result=result)
                write_json(path/'membership.json',dict(method=method,test=fold['test'],validation=fold['validation'],train=list(fold['train'])))
                fits.append(record); results.append(result)
            for a,b in zip(results[0]['history'],results[1]['history']):
                if a['steps']==b['steps'] and a['schedule_hash']!=b['schedule_hash']:
                    raise ValueError('Matched crop schedules diverged')
        freeze_manifest([p for p in (run/'fitting').rglob('*') if p.is_file()],run/'fits-manifest.json')
        write_json(run/'fit-complete.json',dict(fits=fits,elapsed_seconds=time.monotonic()-start))
    elif args.phase=='evaluate':
        if not (run/'fit-complete.json').exists(): raise ValueError('All fits must be frozen before evaluation')
        verify_manifest(run/'fits-manifest.json')
        (run/'predictions').mkdir(exist_ok=False); (run/'panels').mkdir()
        scores={m:{} for m in ('R0','R1','R2')}; displays={}
        for method in config['methods']: (run/'predictions'/method).mkdir()
        for sid in STACK_IDS:
            x,y=load_stack(root,sid); normalized=normalize_stack(x)[0]
            predictions={'R0':tifffile.imread(reference/'held-out'/'unet'/f'{sid}_prediction.tiff')!=0}
            for method,channels in config['methods'].items():
                folder=run/'fitting'/f'{method}-{sid}'
                selection=json.loads((folder/'selection.json').read_text())['selected']
                probabilities=predict_stack(normalized,load_predictor(folder/'best.pt',device_name()),channels,
                                            config['training']['patch_size'],config['training']['stride'])
                mask=probabilities>=selection['threshold']
                if mask.shape!=x.shape: raise ValueError('Prediction shape differs from source')
                predictions[method]=mask
                tifffile.imwrite(run/'predictions'/method/f'{sid}.tiff',mask.astype(np.uint8)*255,photometric='minisblack')
            metrics=comparison_metrics(y,predictions)
            for method,r in metrics.items():
                scores[method][sid]=r
                write_panels(x,y,predictions[method],run/'panels'/f'{method}-{sid}.png')
            axial_panel(x,y,predictions,data['locations'][sid],run/'panels'/f'axial-{sid}.png')
            displays[sid]=(x,y,predictions)
            print('EVALUATED',sid,{m:r['dice'] for m,r in metrics.items()},flush=True)
        write_json(run/'metrics.json',scores)
        overview(displays,run/'comparison-overview.png')
    else:
        verify_manifest(run/'fits-manifest.json')
        print(json.dumps(make_report(run),indent=2))


if __name__=='__main__': main()
