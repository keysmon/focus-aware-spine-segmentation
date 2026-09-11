# ECE435 Binary Segmentation

This project compares binary segmentation of four labeled TIFF stacks (`2`, `3`, `6`, `7`). The original `project.py` and exploratory notebooks are preserved. The reproducible comparison is in `segmentation/`.

## Final Development Result

The strongest evaluated ensemble achieved **73.35% macro foreground F1/Dice**, with **71.35% precision** and **76.94% recall**, averaged equally across four stacks. The 80% target was not reached. These are repeated development comparisons, not independent test-set estimates.

The commands below reproduce the baseline and focus-aware workflows; they do not recreate the final ensemble automatically. Experimental model components are in `experiments/`. Training checkpoints, generated runs, and local development notes are not distributed. See [Results and limitations](docs/results.md).

## Setup

```bash
UV_CACHE_DIR=/tmp/ece435-uv-cache uv venv --python 3.12 .venv
UV_CACHE_DIR=/tmp/ece435-uv-cache uv pip install --python .venv/bin/python -r requirements-lock.txt
```

`requirements-lock.txt` records the environment used for this experiment (macOS ARM64). For another platform, resolve `requirements-dev.txt` and record its versions. No paid compute is required. Original notebooks may additionally require Jupyter, torchvision, and torchsummary; they are not needed by the comparison CLI.

## Run

Run from this directory; use a new run ID for each experiment.

```bash
MPLCONFIGDIR=/tmp/ece435-mpl .venv/bin/python -m pytest tests -q
MPLCONFIGDIR=/tmp/ece435-mpl .venv/bin/python -m segmentation audit --data . --run runs/comparison-002
MPLCONFIGDIR=/tmp/ece435-mpl .venv/bin/python -m segmentation fit --config configs/comparison.json --run runs/comparison-002
MPLCONFIGDIR=/tmp/ece435-mpl .venv/bin/python -m segmentation evaluate --run runs/comparison-002
MPLCONFIGDIR=/tmp/ece435-mpl .venv/bin/python -m segmentation report --run runs/comparison-002
```

Inspect the audit panels for alignment before fitting. `audit` validates inputs and runs the original baseline. `fit` performs validation-only classical selection and trains one fresh U-Net per fold. `evaluate` requires all four frozen selections and unchanged configuration/source/input hashes. `report` creates a comparison table and decision summary. Duplicate run IDs and completed phases are not overwritten; interrupted runs retain partial artifacts and require a new run for a clean restart.

## Methods and Evaluation

- C0: original OpenCV pipeline.
- C1: per-slice percentile normalization before the baseline.
- C2: C1 with an adaptive final density threshold.
- C3: C2 plus closing and small-component removal.
- Small 2D U-Net: paired augmented 128×128 training patches, BCE plus Dice loss, overlap-averaged full-resolution inference.

Each fold uses two training stacks, one validation stack, and one test stack. Splits are frozen in `segmentation/splits.py`; candidate settings and training caps are in `configs/comparison.json`. CPU uses two PyTorch threads; MPS is used when available. Training stops at 20 epochs, five stale validation epochs, or 15 minutes of training per fold. A synthetic overfit check precedes real training.

Report Dice/F1, precision, recall, per-stack macro averages, and symmetric boundary distances in pixels. Accuracy is secondary because only about 1–2% of pixels are foreground. Both-empty masks have Dice 1; one-empty masks have Dice 0. Undefined precision/recall are null. Boundary distances exclude empty-boundary slices and explicitly count those cases.

## Outputs and Limits

`runs/<id>/` contains audit metadata and original-file hashes, frozen configuration/source hashes, installed dependency versions, baseline and held-out TIFF masks, comparison panels, checkpoints, selection histories, metrics, and `report.md`. Red overlay pixels are false positives; blue are false negatives; green are true positives.

Only four stacks are available. Their acquisition independence and the historical pipeline's tuning provenance are unverified. Initial label inspection is documented; this is a limited comparison, not a claim of broad or clinical reliability. A candidate is not promoted merely for higher average accuracy. Preserve inputs and previous runs.

See [Results and limitations](docs/results.md) for the final development results.

## Focus-Aware Follow-Up

The labels mark **in-focus foreground**, not all visible structures. The focus-aware experiment compares saved first-run predictions (R0), a revised single-slice model (R1), and a matched five-slice model (R2). The new package is `segmentation/focus/`; the original implementation is unchanged.

```bash
MPLCONFIGDIR=/tmp/ece435-mpl .venv/bin/python -m segmentation.focus audit --run runs/focus-comparison-002 --reference runs/comparison-002
MPLCONFIGDIR=/tmp/ece435-mpl .venv/bin/python -m segmentation.focus fit --run runs/focus-comparison-002
MPLCONFIGDIR=/tmp/ece435-mpl .venv/bin/python -m segmentation.focus evaluate --run runs/focus-comparison-002
MPLCONFIGDIR=/tmp/ece435-mpl .venv/bin/python -m segmentation.focus report --run runs/focus-comparison-002
```

Complete all four initial comparison commands first. Use an unused run ID and pass that completed comparison with `--reference`, as shown above. Audit uses `configs/focus-comparison.json`. Inspect `audit-panels/` before fitting. Fit freezes all eight model selections before evaluation. Complete focal sweeps are required for normalization; this is not a streaming predictor.

R1/R2 use identical sampled crop sequences, stack-wide percentile scaling, and shared channel augmentations. Sampling mixes foreground, high-gradient negative pixels, and uniform centers. Negative-centered crops retain any real foreground in the crop. R2 uses offsets -2 to +2 and predicts only the middle frame, repeating edge frames when necessary.

Each fit is capped at 20 epochs or 15 minutes of training, with early stopping; eight fits have a maximum of 120 training minutes. The overall fitting deadline is three hours including validation. Partial runs are preserved but are not automatically resumable. No paid compute is used.

Additional results include empty-slice activation, empty-slice false-positive pixel rates, per-slice scores, and shared-subset boundary distances. Report R2−R1 separately from R1−R0. These four stacks have already influenced development, so the follow-up is development cross-validation rather than independent confirmation. See [Results and limitations](docs/results.md).
