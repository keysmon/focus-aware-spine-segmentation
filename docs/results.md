# Results and limitations

## Task and evaluation

Segment hand-annotated in-focus foreground in four TIFF stacks (IDs 2, 3, 6, 7). White mask pixels are positive; dark pixels include background and out-of-focus structures. Foreground occupies roughly 1–2% of pixels.

Each fold uses two training stacks, one validation stack, and one evaluation stack. Settings are selected on validation data. Foreground F1 equals Dice: `2TP / (2TP + FP + FN)`. Macro metrics average the four stack-level scores equally; they are not pooled-pixel scores.

## Best completed comparison

| Method | Macro F1/Dice | Macro precision | Macro recall |
|---|---:|---:|---:|
| Single-slice reference (R1) | 71.6885% | 70.4057% | 74.6601% |
| Three-model probability ensemble | 73.3463% | 71.3459% | 76.9364% |

The ensemble averages probabilities from a BCE+Tversky U-Net, an attention U-Net, and a short axial 3D model. It improved macro F1 by 1.66 percentage points over R1, but did not reach the 80% target. Its mean boundary distance on the within-comparison shared subset was 7.60 pixels versus 6.92 for R1; overlap improvement did not imply better boundaries.

## Other approaches

Experiments explored black-hat morphology, closing, opening, connected-component filtering, Gaussian and bilateral filters, local contrast, focus and Hessian features, Gabor filters, graph-cut refinement, tree classifiers, neural architectures, losses, augmentation, transfer learning and ensembles. Several improved a validation stack without improving the four-stack comparison. Recent MobileNetV3, sharpness-aware minimization and local-standardization screens did not beat the attention controls.

## Scope of this release

The repository includes paired inputs, comparison CLIs, model components, configuration and behavioral tests. Generated runs, fitted checkpoints, per-run scripts and local planning records are excluded. The checked-in CLIs implement the baseline and focus-aware comparisons, not a one-command reproduction of the final ensemble. The table records the completed local experiment; released weights are not available for independent replay.

Only four stacks were available and repeatedly informed development. These scores do not establish performance on independent acquisitions. Validation-only and label-assisted diagnostic scores must not be presented as four-stack results. Additional untouched data would be needed to assess generalization.
