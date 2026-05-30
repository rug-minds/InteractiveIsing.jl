# Baseline Best-Case Runtime Diagnostic

- Date: 2026-05-30
- Architecture: `784 -> 120 -> 40`
- Script: `bestcase_256_runtime.jl`
- Measured examples: `256`
- Warmup: one extra sample before timing, excluded from measured totals.
- Batch structure: two measured minibatches of `128` examples.
- Dynamics/work: same baseline free and nudged input-field process algorithm, with `sweeps = 500`, `beta = 5.0`, `lr = 0.0015`, `weight_decay = 0.0`.

## Purpose

This is a best-case runtime diagnostic for the baseline architecture. It keeps the same per-sample free/nudged work and gradient/update work, but bypasses `ProcessManager` dispatch by running persistent workers directly with a static thread partition.

Outputs are written to `summary.csv`.
