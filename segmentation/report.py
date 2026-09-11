"""Immutable experiment artifacts and identical comparison panels."""
import json
import os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ece435-mpl')
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from segmentation.data import digest
from segmentation.splits import FOLDS


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def create_run(output: Path):
    output.mkdir(parents=True, exist_ok=False)


def verify_inputs(root, hashes):
    for name, expected in hashes.items():
        if digest(Path(root) / name) != expected:
            raise ValueError(f'Input changed: {name}')


def freeze_config(config, destination):
    expected = [dict(f, train=list(f['train'])) for f in FOLDS]
    if config.get('folds') != expected:
        raise ValueError('Invalid fold definitions')
    write_json(destination, config)


def write_panels(images, reference, prediction, destination: Path):
    indices = sorted(set(np.linspace(0, len(images)-1, min(5, len(images)), dtype=int).tolist() +
                         [int(reference.sum(axis=(1,2)).argmax())]))
    fig, axes = plt.subplots(len(indices), 4, figsize=(9, 3*len(indices)), squeeze=False)
    for row, i in enumerate(indices):
        axes[row, 0].imshow(images[i], cmap='gray')
        axes[row, 1].imshow(reference[i], cmap='gray', vmin=0, vmax=1)
        axes[row, 2].imshow(prediction[i], cmap='gray', vmin=0, vmax=1)
        overlay = np.zeros((*reference[i].shape, 3))
        overlay[..., 0] = prediction[i] & ~reference[i]
        overlay[..., 1] = prediction[i] & reference[i]
        overlay[..., 2] = reference[i] & ~prediction[i]
        axes[row, 3].imshow(overlay)
        for ax, title in zip(axes[row], ['Image', 'Reference', 'Prediction', 'FP red / FN blue / TP green']):
            ax.set_title(f'Slice {i}: {title}', fontsize=8); ax.axis('off')
    fig.tight_layout(); fig.savefig(destination, dpi=100); plt.close(fig)


def summarize(run):
    run = Path(run)
    records = json.loads((run/'metrics.json').read_text())
    rows = ['# Segmentation Comparison', '', '| Method | Stack | Dice/F1 | Precision | Recall | Boundary mean px |',
            '| --- | --- | --- | --- | --- | --- |']
    def fmt(value): return 'N/A' if value is None else f'{value:.4f}'
    summary = {}
    for method, stacks in records.items():
        summary[method] = {}
        for sid, metrics in stacks.items():
            rows.append('| ' + ' | '.join([method, sid] + [fmt(metrics[k]) for k in
                ['dice', 'precision', 'recall', 'boundary_mean_px']]) + ' |')
        for key in ('dice','precision','recall','boundary_mean_px','boundary_p95_px'):
            values = [m[key] for m in stacks.values() if m[key] is not None]
            summary[method][key] = float(np.mean(values)) if values else None
            summary[method][key+'_available_stacks'] = len(values)
        rows.append('| '+ ' | '.join([method, 'Macro']+[fmt(summary[method][k]) for k in
            ['dice','precision','recall','boundary_mean_px']])+' |')
    rows += ['', '## Decision', '']
    fit_status = json.loads((run/'fit-complete.json').read_text())
    timed_out = any(f['neural']['status'] == 'time_limit' for f in fit_status['folds'])
    eligible = []
    for method in ('classical','unet'):
        improved = sum(records[method][sid]['dice'] > records['baseline'][sid]['dice'] for sid in records['baseline'])
        if improved >= 3 and summary[method]['dice'] > summary['baseline']['dice']:
            eligible.append(method)
        rows.append(f'- {method}: higher Dice on {improved}/4 stacks; inspect precision, recall and boundary tradeoffs above.')
    rows.append('')
    if timed_out:
        rows.append('Training reached its time cap. These are provisional checkpoint results; no final winner is established. Retain the baseline pending a separately designed experiment.')
    elif eligible:
        winner = max(eligible, key=lambda name: summary[name]['dice'])
        rows.append(f'{winner} satisfies the predefined Dice consistency criterion. Review its per-stack tradeoffs before adoption; the historical baseline remains preserved.')
    else:
        rows.append('Neither candidate satisfies the predefined consistency criterion. Retain the baseline; this experiment does not demonstrate a reliable replacement.')
    rows += ['', '## Interpretation and Limitations', '',
        'Boundary distances summarize slices with both boundaries present; metrics.json records eligible, one-empty and both-empty slice counts, confusion counts, accuracy, and undefined precision/recall availability. Missing values are not zero.',
        'Foreground is sparse. Accuracy is secondary. All stack scores and macro averages are shown; no confidence or clinical claims are supported.',
        'Only four stacks are available; acquisition independence and historical baseline tuning are unverified. Initial label inspection and cross-fold selection limit claims of untouched evaluation. Method selection does not create a new independent final test set.',
        'Validation settings were frozen before held-out candidate evaluation. Panels use the same slices for every method. Predictions are full-resolution binary TIFF stacks.',
        '', '## Runtime', '', f'Fit elapsed seconds: {fit_status["elapsed_seconds"]:.1f}.',
        *[f'- Test stack {f["test"]}: training {f["neural"]["training_seconds"]:.1f}s, stop={f["neural"]["status"]}, selected epoch={f["neural"]["selected"]["epoch"]}.' for f in fit_status['folds']]]
    write_json(run/'summary.json', summary)
    with (run/'report.md').open('x') as stream: stream.write('\n'.join(rows)+'\n')
    return summary
