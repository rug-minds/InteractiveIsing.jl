# Reduced Baseline Benchmark Refresh

This folder collects the corrected `2026-05-30` reruns after removing the
structural input layer from the `784-120-40` field-input baseline.

## Main Results

- `local_langevin_learning_vs_process.csv`:
  full reduced-graph `LocalLangevin` learning minibatch comparison.
  This is the main apples-to-apples bespoke-versus-`Process` result for the
  current baseline.
- `serial_process_real_params.csv`:
  serial `Process` timing on the real training path.
- `chunk_size_grid_16_32.csv`:
  manager throughput scaling for the real training path with fixed chunk sizes.
- `metropolis_process_vs_bespoke_single_example.csv`:
  reduced-graph Metropolis sanity check. Useful for checking `Process`
  framework overhead separately from the full trainer, but not the main
  learning-loop benchmark.

## Current Takeaways

- Full learning loop, single-thread serial:
  direct bespoke is faster than serial `Process`, but by about `1.4x`, not
  `3x`.
- Real manager throughput:
  the best measured 32-worker point is much faster than serial `Process`.
- Old pre-fix `784-120-40` runtime rows from the structural-input-layer
  baseline should not be used for current conclusions.
