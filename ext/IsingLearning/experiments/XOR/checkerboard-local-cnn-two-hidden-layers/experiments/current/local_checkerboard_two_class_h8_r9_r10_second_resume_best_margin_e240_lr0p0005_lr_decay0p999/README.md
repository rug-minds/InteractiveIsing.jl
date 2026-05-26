# Checkerboard Two-Class H8 R9/R10 Second Resume

This run is a second continuation of the local checkerboard XOR experiment for
radii 9 and 10. It uses the same `8x8 -> 8x8 -> 8x8 -> 4x4` architecture as the
earlier H8 sweep, keeps non-periodic hidden layers, and uses the replicated
two-class output code.

The purpose was to test whether the larger-radius runs could improve or become
more stable after resuming from best-margin checkpoints with a smaller Adam
learning rate.

## Optimizer Settings

The folder name spells out the shorthand that used to be `decay999`:

```text
lr0p0005       -> initial Adam learning rate 0.0005
lr_decay0p999  -> learning-rate multiplier 0.999 per update
```

This is optimizer learning-rate decay, not weight decay. The script also uses a
separate `weight_decay` field for L2 decay on couplings.

## Results

The run compares:

- `h8_r9`: hidden side 8, radius 9, non-periodic.
- `h8_r10`: hidden side 8, radius 10, non-periodic.

Best retained metrics from `summary.csv`:

```text
h8_r9   best_mse=0.11857444  best_min_margin=1.232935   best_margin_epoch=210
h8_r10  best_mse=0.07788126  best_min_margin=1.1733981  best_margin_epoch=110
```

Both configurations reached `accuracy = 1.0`. Radius 10 had the lower best MSE;
radius 9 had the slightly stronger best margin in this continuation.

## Artifacts

- `metrics.csv`: aggregate metrics for both radii.
- `summary.csv`: one-row summary per radius.
- `metrics.png`, `summary.png`: plots generated from the CSV data.
- `h8_r9`, `h8_r10`: per-radius metrics, checkpoints, and parameter files.
- `validation_1024_best_margin_params.*`: high-repeat validation of the
  best-margin checkpoint.
