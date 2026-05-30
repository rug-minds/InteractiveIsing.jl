# R8 Relaxation/Beta Grid

Purpose: test how many Metropolis relaxation sweeps are needed for the
single-hidden local MNIST architecture to learn from scratch at `r=8`.

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
- optimizer: `adam`
- learning rates W0/W12/W2O/B: defaults from `mnist_local_manager_grid.jl`
- gradient normalization: `mean`
- checkpoint format: includes sparse `J`, bias, optimizer state, update index,
  config, source RNG, epoch, and best accuracy

## Grid

Relaxation sweeps use the same value for free and nudged phases.

| free/nudge sweeps | beta |
| ---: | ---: |
| 10 | 2.5 |
| 10 | 5.0 |
| 10 | 10.0 |
| 25 | 2.5 |
| 25 | 5.0 |
| 25 | 10.0 |
| 50 | 2.5 |
| 50 | 5.0 |
| 50 | 10.0 |
| 100 | 2.5 |
| 100 | 5.0 |
| 100 | 10.0 |

## Output

Each grid point writes to a child folder named:

`r8_s<SWEEPS>_beta<BETA>_e30`

Each run folder should contain:

- `settings.md`
- `metrics.csv`
- `progress.png`
- `diagnostics/epoch_time.png`
- `best_params.bin`
- `latest_checkpoint.bin`
- `final_params.bin`

The root launcher also appends a `grid_summary.csv` row after each completed
run.
