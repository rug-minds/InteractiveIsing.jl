# Checkerboard Two-Class Zero-Init H8 R8/R9

This is the short zero-initialized checkerboard run for radii 8 and 9. It used
the same local architecture as the continuation runs:

```text
8x8 checkerboard input -> 8x8 hidden -> 8x8 hidden -> 4x4 two-class output
```

The run used 64 repeats per XOR case, 20 free/nudged sweeps, and 80 epochs. It
is useful mainly as the baseline before the later best-margin continuation
runs.

## Results

Best retained metrics from `summary.csv`:

```text
h8_r8  best_mse=0.044284396  best_min_margin=0.63584864   first_all_correct_epoch=25
h8_r9  best_mse=0.33636203   best_min_margin=-0.03564602  first_all_correct_epoch=-1
```

Radius 8 learned the task. Radius 9 did not reach all-correct classification in
this short zero-init run, which motivated the later resumed runs.

## Artifacts

- `metrics.csv`: aggregate per-epoch metrics for both radii.
- `summary.csv`: one-row summary per radius.
- `metrics.png`, `summary.png`: plots generated from the CSV data.
- `h8_r8`, `h8_r9`: per-radius metrics, checkpoints, and parameter files.
- `validation_h8_r8_512.md`: validation note for the radius-8 checkpoint.
