# R1-R9 Relaxation Grid

Purpose: compare locality radius `r=1:9` at a small set of fixed Metropolis
relaxation settings.

This experiment is queued to run after the active `r=8` relaxation/beta grid so
the two 32-thread jobs do not compete for the same cores.

All runs start from random initialization. No checkpoint resume is used.

## Shared Parameters

- architecture: `784 -> 28x28 -> 11x11 -> 40`
- input handling: field-based input, static hidden/output proposal set
- dynamics: `metropolis`
- epochs: `30`
- workers: `32`
- batch size: `32`
- train/test per class: `100` / `20`
- free/nudge reads: `3` / `3`
- beta: `5.0`
- optimizer: `adam`
- gradient normalization: `mean`
- checkpoint format: includes sparse `J`, bias, optimizer state, update index,
  config, source RNG, epoch, and best accuracy

## Grid

The default grid is intentionally small:

- free/nudge sweeps: `25`, `50`
- radii: `1,2,3,4,5,6,7,8,9`

The launcher supports overrides:

- `ISING_MNIST_RADIUS_SWEEPS`, comma-separated, default `25,50`
- `ISING_MNIST_RADIUS_GRID`, comma-separated, default `1,2,3,4,5,6,7,8,9`

## Folder Layout

Runs are grouped by relaxation steps first, then by radius:

```text
s25/
  r1_e30/
  r2_e30/
  ...
s50/
  r1_e30/
  r2_e30/
  ...
```

Each run folder should contain:

- `settings.md`
- `metrics.csv`
- `progress.png`
- `diagnostics/epoch_time.png`
- `best_params.bin`
- `latest_checkpoint.bin`
- `final_params.bin`

The root launcher appends `grid_summary.csv` after each completed run.
