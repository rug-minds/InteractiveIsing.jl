# XOR Experiment Instructions

Use of this file: operational rules for future agents before running or editing XOR experiments. Keep this as instructions, not results.

- Do diagnostic runs before broad grids. If the metrics are not learning, stop and inspect instead of burning compute.
- Use `ProcessManager` for the new XOR experiments.
- Use Adam by default for XOR experiments. Only use SGD/descent when explicitly comparing optimizers.
- For 32 workers, schedule 32 manager jobs per epoch so work can actually run in parallel.
- Use multiple random initializations/repeats per XOR case.
- Do not schedule one job per random init. Put a few random initializations/repeats inside each worker `ProcessAlgorithm` execution and average them in that worker-local buffer.
- For the 4 XOR cases with 32 workers, default to 8 jobs per case. With 32 repeats per case this means 32 jobs per epoch and each job runs 4 repeats internally.
- If increasing repeats for less noisy gradients, keep at least 32 jobs per epoch and increase repeats per job; e.g. 128 repeats per case gives 32 jobs with 16 repeats per job.
- Share static model data across workers where possible, especially `J`/adjacency and base parameter arrays. Worker state and clamp buffers stay worker-local.
- Use `Optimisers.jl` for adaptive updates when learning-rate mismatch is likely. Adam is the default unless a run explicitly tests SGD/descent.
- Prefer `Optimisers.adjust!` for learning-rate schedules so Adam moment estimates are preserved.
- Include explicit coupling weight decay in noisy contrastive XOR runs; start with `1e-4` when copying the successful multiplexed-pattern recipe.
- When LocalLangevin-only runs bounce, compare with `BlockLangevin` in the same manager file instead of changing the manager path.
- Accuracy alone is not enough. Log per-case predictions, scores, margins, MSE, and whether all four cases are correct.
- Best-MSE bar plots without stable learning curves or seed robustness are not useful results.
- CSVs alone are not a complete result. If a run folder has `metrics.csv` or `summary.csv` without matching PNGs, run `plot_run_results.jl` before calling it done.
- For the local checkerboard XOR architecture, compare hidden sizes `8x8` and `16x16` and local NN/radius up to `10`.
- If majority-vote output only gives transient tiny-margin correctness, use the replicated `two_class` output before doing another broad grid; `pattern` is useful as a diagnostic but was not robust here.
- When high-repeat validation rejects logged best checkpoints, test continuous zero initialization before sweeping more hyperparameters. Training and validation init modes must match.
- For the local CNN-like checkerboard file, the first robust recipe is `output_mode=two_class`, `init_mode=zero`, `hidden=8x8`, `radius=8`, `BlockLangevin`, 20 free/nudged sweeps, 32 jobs per epoch, Adam, and high-repeat validation of `best_margin_params.bin`.
- Do not treat nearby radii as interchangeable. In the first zero-start two-class run, radius 8 solved with a wide margin while radius 9 still missed one case by epoch 80.
- For the edge-input file, use the replicated `two_class` output as the default. The old scalar majority edge readout is still an option, but it was not the solved recipe.
- For edge-input diagnostics, start with `side=16`, `init_mode=zero`, `dynamics=block`, `free/nudged=20`, `β=1`, Adam `lr=0.002`, decay `0.995`, weight decay `1e-4`, 64 repeats per case, and 32 jobs per epoch.
- In the first clean edge-input sweep, NN `6`, `7`, and `9` solved robustly; NN `1-4` did not. Validate promising edge checkpoints with at least 512 evaluation repeats before treating them as useful.
- Keep learning history, final params, best params, best-margin params, and snapshots when requested.
- Update this file whenever a structural experiment mistake is found.
