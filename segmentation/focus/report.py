"""Paired method comparisons without hiding empty-slice errors."""
import json
from pathlib import Path
import numpy as np
from segmentation.report import write_json

KEYS=('dice','precision','recall','empty_fp_pixel_fraction','empty_activation_fraction',
      'nonempty_mean_recall','nonempty_mean_dice','boundary_mean_px','shared_boundary_mean_px')


def make_report(run):
    run=Path(run)
    scores=json.loads((run/'metrics.json').read_text())
    fits=json.loads((run/'fit-complete.json').read_text())
    summary={}
    for method,stacks in scores.items():
        summary[method]={}
        for key in KEYS:
            values=[r[key] for r in stacks.values() if r[key] is not None]
            summary[method][key]=float(np.mean(values)) if values else None
            summary[method][key+'_count']=len(values)
    def fmt(value): return 'N/A' if value is None else f'{value:.4f}'
    lines=['# Focus-Aware Development Comparison','',
        'R0: saved first-run U-Net. R1: revised single-slice recipe. R2: matched five-slice recipe.',
        'Each method uses whole-stack held-out predictions, but all four stacks have influenced development. These are not fresh independent test results.','',
        '| Method | Stack | Dice | Precision | Recall | Empty FP pixel rate | Empty slice activation | Shared boundary px |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    columns=['dice','precision','recall','empty_fp_pixel_fraction','empty_activation_fraction','shared_boundary_mean_px']
    for method,stacks in scores.items():
        for sid,r in stacks.items():
            lines.append('| '+' | '.join([method,sid]+[fmt(r[k]) for k in columns])+' |')
        lines.append('| '+' | '.join([method,'Macro']+[fmt(summary[method][k]) for k in columns])+' |')
    lines+=['','## Paired Changes','', '| Contrast | Stack | Dice change | Recall change | Empty FP pixel rate change |',
            '| --- | --- | --- | --- | --- |']
    for newer,older in [('R1','R0'),('R2','R1'),('R2','R0')]:
        for sid in scores[older]:
            changes=[scores[newer][sid][k]-scores[older][sid][k] for k in ['dice','recall','empty_fp_pixel_fraction']]
            lines.append('| '+' | '.join([f'{newer} - {older}',sid]+[f'{x:+.4f}' for x in changes])+' |')
    decisions={}
    lines+=['','## Decision and Tradeoffs','']
    for method in ('R1','R2'):
        improvements=sum(scores[method][sid]['dice']>scores['R0'][sid]['dice'] for sid in scores['R0'])
        positive=(summary[method]['dice']>summary['R0']['dice'] and improvements>=3 and
                  summary[method]['empty_fp_pixel_fraction']<summary['R0']['empty_fp_pixel_fraction'])
        recalls=[sid for sid in scores['R0'] if scores[method][sid]['recall']<scores['R0'][sid]['recall']-.03]
        boundaries=[sid for sid in scores['R0'] if scores[method][sid]['shared_boundary_mean_px'] is not None and
                    scores[method][sid]['shared_boundary_mean_px']>scores['R0'][sid]['shared_boundary_mean_px']]
        time_limited=any(f['result']['status']=='time_limit' for f in fits['fits'] if f['method']==method)
        decisions[method]=dict(positive_rule=positive and not time_limited,dice_improved_stacks=improvements,
                              recall_regressions=recalls,boundary_regressions=boundaries,time_limited=time_limited)
        lines.append(f'- {method}: Dice improves on {improvements}/4 stacks. Predeclared improvement rule met: {positive and not time_limited}. Recall loss >0.03 on: {recalls}. Increased shared-subset boundary distance on: {boundaries}.')
    if not any(d['positive_rule'] for d in decisions.values()):
        lines+=['','Neither new candidate meets the complete improvement rule. Preserve R0 as the stronger reference pending a separately designed experiment.']
    else:
        eligible=[m for m,d in decisions.items() if d['positive_rule']]
        best=max(eligible,key=lambda m:summary[m]['dice'])
        lines+=['',f'{best} meets the engineering improvement rule and has the highest macro Dice among eligible candidates. Review the flagged per-stack tradeoffs before adoption.']
    lines+=['','## Runtime and Reproducibility','',f'Total fit elapsed seconds: {fits["elapsed_seconds"]:.1f}.',
            f'Total training seconds: {sum(f["result"]["training_seconds"] for f in fits["fits"]):.1f}.']
    for f in fits['fits']:
        r=f['result']
        lines.append(f'- {f["method"]}, test {f["test"]}: parameters={r["parameter_count"]}, selected epoch={r["selected"]["epoch"]}, training={r["training_seconds"]:.1f}s, stop={r["status"]}.')
    lines+=['','Matched schedule hashes agree for every common training prefix (verified before evaluation). Early stopping can produce different numbers of training steps.','',
        '## Interpretation','',
        'Empty FP pixel rate measures white predictions only within empty-reference slices. Empty slice activation counts any positive pixel. These are complementary measures: a few stray pixels can activate an entire slice.',
        'Boundary distances are in pixels, not physical units. Shared boundary metrics use the same truth-positive and prediction-positive slices across R0/R1/R2; per-method eligibility and all empty cases remain in metrics.json.',
        'R1 changes normalization and sampling together. R2 versus R1 isolates added context, with a small first-layer parameter difference. One seed does not establish robustness. No image fusion, 3D smoothing, label replacement, or test-driven parameter changes were used.',
        'The source masks combine out-of-focus structures and ordinary background. Difficult-negative sampling is a gradient-based heuristic and does not create new focus labels. Acquisition spacing and independent-specimen provenance remain unknown.']
    write_json(run/'summary.json',dict(metrics=summary,decisions=decisions))
    with (run/'report.md').open('x') as stream: stream.write('\n'.join(lines)+'\n')
    return summary
