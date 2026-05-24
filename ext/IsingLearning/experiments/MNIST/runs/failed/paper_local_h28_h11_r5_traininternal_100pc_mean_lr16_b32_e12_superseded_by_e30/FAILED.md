# Superseded Run: LR16 E12

Use of this file: explains why this learned run is under `failed`.

This was the first corrected local paper-manager run with 32-sample minibatches, `gradient_normalization = :mean`, and a 16x LR scale. It reached best small-test accuracy `0.575` after 12 epochs, but the same recipe was rerun for 30 epochs and reached a materially better checkpoint. The 12-epoch run is kept only as a diagnostic showing that the recipe was learning.
