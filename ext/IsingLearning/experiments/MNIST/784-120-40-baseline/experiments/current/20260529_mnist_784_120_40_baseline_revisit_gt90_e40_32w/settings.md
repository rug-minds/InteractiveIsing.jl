# 784-120-40 Baseline Revisit

Purpose: revisit the baseline that previously reached `0.8852` test accuracy
and test whether gentler Adam settings can push it above `0.90`.

All runs start from random initialization. No checkpoint resume is used.

## Code State

The baseline has not received the full single-hidden-local rewrite:

- it still uses worker-local second `MagField` input storage;
- it still uses the baseline `LocalLangevin` process stack;
- it does inherit generic repo-wide process/sparse fixes.

Safe baseline-specific fixes applied before this run:

- explicit `@bind` / `@merge` state sharing in the worker composite;
- CairoMakie lazy-load world-age fix.

## Shared Parameters

- architecture: `784 -> 120 -> 40`
- workers: `32`
- batch size: `128`
- train/test per class: `5421` / `892`
- train eval per class: `100`
- sweeps: `500`
- relaxation steps: `80000`
- temperature: `0.001`
- Langevin stepsize: `0.5`
- weight scale: `0.005`
- eval every: `5`
- epochs: `40`
- optimizer: Adam

## Grid

| name | beta | lr | weight decay |
| --- | ---: | ---: | ---: |
| `beta5_lr0020_wd0_e40` | 5.0 | 0.0020 | 0.0 |
| `beta5_lr0015_wd0_e40` | 5.0 | 0.0015 | 0.0 |
| `beta3_lr0020_wd0_e40` | 3.0 | 0.0020 | 0.0 |

The previous best run used `beta=5.0`, `lr=0.003`, and peaked at epoch `15`
with test accuracy `0.885201793721973`, then degraded.

## Queue

This grid is queued after the single-hidden local experiment chain finishes.
The queue watcher waits for the high-sweep r8 grid to write 3 completed rows.
