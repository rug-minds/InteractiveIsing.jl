# Failed Run: Paper Local H28-H11 R5 Train-Internal 100/Class

Use of this file: explain why this failed run was kept instead of deleted.

This run was stopped early because it collapsed to one-class predictions almost immediately:

- architecture: `784 field -> 28x28 -> 11x11 -> 40`
- radius: `5`
- trainable same-layer couplings: enabled
- train/test per class: `100 / 50`
- workers/batch: `32 / 64`
- reads: `3 / 3`
- sweeps: `50 / 50`
- learning rates: `W0/W12/W2O=0.003`, `W11/W22/WOO=0.0005`, `B=0.0003`
- stopped after epoch `4`

Reason kept: the old notes reported this family can work, but this clean-script run with these settings collapsed to predicting digit 0 for every test sample by epoch 1. Keep it as a warning that this exact from-scratch recipe/seed is not a good curated run.
