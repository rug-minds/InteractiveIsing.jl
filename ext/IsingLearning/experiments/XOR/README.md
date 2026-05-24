# XOR Experiments

This folder is the clean starting point for new shareable learning runs. The
training files default to `32` `ProcessManager` workers; start Julia with
`-t 32` for the intended run shape.

The archived exploratory folders were moved to:

```text
ext/IsingLearning/ExperimentsOld
```

## Files

- `xor_process_baseline.jl`: small all-to-all XOR baseline using
  `ProcessManager`, `LayeredIsingGraphLayer`, worker-local input averaging,
  and the current contrastive learning API.
- `xor_manager_input_averaging.jl`: clean `2 -> 4 -> 2xR` XOR demonstrator
  using `ProcessManager`, the standard `LayerContrastiveStep`, and worker-local
  repeat averaging before a single `FlushAtEnd()` gradient merge.
- `xor_local_cnn_like_grid.jl`: local CNN-like checkerboard XOR comparison for
  `8x8 -> HxH -> HxH -> 4x4`, sweeping local radius and hidden-layer
  periodicity with aggregate metrics, snapshots, final parameters, and plots.
  Its defaults use the first robust local recipe: zero-start `BlockLangevin`
  and a replicated two-class output code.
- `xor_edge_application_grid.jl`: edge-application XOR revival. A checkerboard
  input line drives one edge of a `16x16` layer, the opposite edge is read out
  with a replicated two-class line by default, and hidden NN is swept up to 10.
  It uses the same manager-backed zero-start `BlockLangevin` recipe as the
  robust local checkerboard run.
- `plot_run_results.jl`: post-processes every recognized `metrics.csv` and
  `summary.csv` under `runs` and writes PNG plots beside the CSV files.

Run from the repository root:

```powershell
julia -t 32 --project=ext/IsingLearning ext/IsingLearning/experiments/XOR/xor_process_baseline.jl
julia -t 32 --project=ext/IsingLearning ext/IsingLearning/experiments/XOR/xor_manager_input_averaging.jl
julia -t 32 --project=ext/IsingLearning ext/IsingLearning/experiments/XOR/xor_local_cnn_like_grid.jl
julia -t 32 --project=ext/IsingLearning ext/IsingLearning/experiments/XOR/xor_edge_application_grid.jl
julia --project=ext/IsingLearning ext/IsingLearning/experiments/XOR/plot_run_results.jl
```
