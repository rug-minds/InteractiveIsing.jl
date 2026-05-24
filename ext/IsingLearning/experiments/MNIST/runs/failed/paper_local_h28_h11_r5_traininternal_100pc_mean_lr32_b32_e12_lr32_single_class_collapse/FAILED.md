# Failed Run: LR32 Collapse

Use of this file: explains why this run was kept under `failed`.

This tested the same local paper-manager architecture as the LR16 run, but with a 32x mean-normalized LR scale. It collapsed to single-class predictions and stayed near chance accuracy. The run is kept because it brackets the useful minibatch learning-rate range: LR16 learned, LR32 was too aggressive.
