# Focus-aware spine segmentation

Binary segmentation of microscopy TIFF stacks using classical image processing and U-Net models. The masks identify in-focus structures: white is foreground, while black includes background and out-of-focus structures.

The project compares filtering and morphology with single-slice and multi-slice neural models. It includes paired data loading, patch sampling, training, evaluation, and visual reports.

## Setup

Use Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

For the exact macOS ARM64 environment used during development, install `requirements-lock.txt` instead. Training uses Apple MPS when available and otherwise falls back to CPU.

## Run a comparison

The four image/mask pairs are in `data/`. Run these commands from the repository root:

```bash
python -m segmentation audit --run runs/comparison
python -m segmentation fit --config configs/comparison.json --run runs/comparison
python -m segmentation evaluate --run runs/comparison
python -m segmentation report --run runs/comparison
```

The audit checks the data and generates baseline predictions. Inspect its panels before training. Each run saves models, predictions, metrics, and comparison images under `runs/`. Use a new directory for each run; completed output is never overwritten.

To compare single-slice and five-slice models, first complete the comparison above, then run:

```bash
python -m segmentation.focus audit --reference runs/comparison --run runs/focus
python -m segmentation.focus fit --run runs/focus
python -m segmentation.focus evaluate --run runs/focus
python -m segmentation.focus report --run runs/focus
```

Settings are in `configs/`. Both commands accept `--data` for a different input directory. Input files follow the pattern `{id}_image.tiff` and `{id}_mask.tiff`.

## Results

| Method | F1 / Dice | Precision | Recall |
|---|---:|---:|---:|
| Single-slice reference | 71.69% | 70.41% | 74.66% |
| Best ensemble | 73.35% | 71.35% | 76.94% |

Scores are averaged equally across four stacks. Each fold uses two stacks for training, one for validation, and one for evaluation. Foreground covers only about 1–2% of the images, so accuracy alone is not useful here.

The ensemble combines a U-Net trained with Tversky loss, an attention U-Net, and an axial 3D model. Its model components are included, but the trained weights and experiment-specific runner are not. The commands above run the baseline and focus comparisons.

These four stacks were used repeatedly during development. The results need confirmation on new data. See [evaluation notes](docs/results.md) for details.

## Layout

- `data/`: original image and mask pairs
- `segmentation/`: comparison pipeline; `focus/` handles focal context and `models/` contains ensemble components
- `configs/`: training and comparison settings
- `tests/`: loading, sampling, model, metric, and workflow checks

Run the tests with `python -m pytest tests -q`.
