# R8 High-Sweep Relaxation Grid

Purpose: test whether the `r=8` single-hidden local MNIST architecture learns
when Metropolis relaxation is pushed well beyond the first sweep grid.

This experiment is queued after:

1. `20260529_mnist_single_hidden_local_r8_relax_beta_grid_e30_32w`
2. `20260529_mnist_single_hidden_local_r1_to_r9_relax_grid_e30_32w`

All runs start from random initialization. No checkpoint resume is used.

## Shared Parameters

- architecture: `784 -> 28x28 -> 11x11 -> 40`
- input handling: field-based input, static hidden/output proposal set
- dynamics: `metropolis`
- radius: `8`
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

Default free/nudge sweeps:

- `150`
- `250`
- `400`

The launcher supports overrides:

- `ISING_MNIST_HIGH_SWEEPS`, comma-separated, default `150,250,400`
- `ISING_MNIST_HIGH_SWEEP_BETA`, default `5.0`
- `ISING_MNIST_HIGH_SWEEP_RADIUS`, default `8`

## Folder Layout

```text
s150/r8_beta5p0_e30
s250/r8_beta5p0_e30
s400/r8_beta5p0_e30
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
