# Checkerboard Two-Class H8 R7 To R10 Resume

This run continues the local checkerboard XOR experiment from saved best-margin
checkpoints and compares radii 7, 8, 9, and 10 with hidden side length 8.

Architecture:

```text
8x8 checkerboard input -> 8x8 hidden -> 8x8 hidden -> 4x4 two-class output
```

The hidden layers are non-periodic. The output is a replicated two-class readout
for XOR false versus XOR true.

## Optimizer Settings

The old folder name used `decay998`. That means:

```text
lr0p001        -> initial Adam learning rate 0.001
lr_decay0p998  -> learning-rate multiplier 0.998 per update
```

This is learning-rate decay. It is separate from `weight_decay`, which is the L2
penalty on trainable couplings.

## Results

Best retained metrics from `summary.csv`:

```text
h8_r7   best_mse=0.06282423   best_min_margin=0.9554761  best_margin_epoch=40
h8_r8   best_mse=0.12536147   best_min_margin=1.1238952  best_margin_epoch=120
h8_r9   best_mse=0.05217444   best_min_margin=1.2207861  best_margin_epoch=180
h8_r10  best_mse=0.042092927  best_min_margin=1.0790174  best_margin_epoch=200
```

All four radii reached `accuracy = 1.0`. Radius 10 had the lowest best MSE;
radius 9 had the strongest best margin. This is the best retained broad sweep
for the H8 checkerboard architecture.

## Artifacts

- `metrics.csv`: aggregate per-epoch metrics.
- `summary.csv`: one-row summary per radius.
- `metrics.png`, `summary.png`: plots generated from the CSV data.
- `h8_r7`, `h8_r8`, `h8_r9`, `h8_r10`: per-radius metrics, checkpoints, and
  parameter files.
- `validation_1024_best_margin_params.*`: high-repeat validation of the
  best-margin checkpoints.
