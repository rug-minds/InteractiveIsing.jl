# XOR Experiments

This folder indexes the XOR learning experiments. Each runnable experiment lives
in its own architecture-named folder with its script, run data, plots, and any
architecture schematic for that family.

The archived exploratory folders were moved to:

```text
ext/IsingLearning/ExperimentsOld
```

## Experiment Folders

- `two-input-2x2-hidden-majority-vote-baseline`: small `2 -> 2x2 -> 4` all-to-all ProcessManager/Adam baseline with majority-vote output replicas.
- `all-to-all-input-averaged-readout`: direct `2 -> 4 -> 2xR` all-to-all ProcessManager demonstrator with worker-local repeat averaging.
- `checkerboard-local-cnn-two-hidden-layers`: local checkerboard input on an `8x8 -> HxH -> HxH -> 4x4` CNN-like Ising architecture.
- `edge-driven-single-layer-readout`: edge-applied XOR input driving a `16x16` layer with the opposite edge used as the readout.
- `diagnostics`: smoke, sanity, scout, and timing material that is not part of the result experiment trees.

Within architecture folders, `experiments/current/<experiment-name>` contains
the retained successful experiment data, generated per-run PNG plots, and a copy
of the Julia simulation file used for that experiment. Leaf configuration metric
plots are flattened into the parent experiment folder with names like
`h8_r9_metrics.png`. `aggregate_plots` contains cross-experiment comparisons,
and `schematic.png` is the generated architecture figure when that experiment
has one.

## Utilities

- `plot_run_results.jl`: post-processes CSV files under the architecture folders and writes per-run PNGs into `experiments/current`.
- `../plot_architecture_schematics.jl`: regenerates the schematic PNGs inside the architecture folders.

Run from the repository root:

```powershell
julia -t 32 --project=ext/IsingLearning ext/IsingLearning/experiments/XOR/two-input-2x2-hidden-majority-vote-baseline/plotting.jl
julia -t 32 --project=ext/IsingLearning ext/IsingLearning/experiments/XOR/all-to-all-input-averaged-readout/xor_manager_input_averaging.jl
julia -t 32 --project=ext/IsingLearning ext/IsingLearning/experiments/XOR/checkerboard-local-cnn-two-hidden-layers/xor_local_cnn_like_grid.jl
julia -t 32 --project=ext/IsingLearning ext/IsingLearning/experiments/XOR/edge-driven-single-layer-readout/xor_edge_application_grid.jl
julia --project=ext/IsingLearning ext/IsingLearning/experiments/XOR/plot_run_results.jl
julia --project=ext/IsingLearning ext/IsingLearning/experiments/plot_architecture_schematics.jl
```
