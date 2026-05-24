# Local Checkerboard XOR Run

Inputs are two physical checkerboard freeze masks. No four-case one-hot input code is used.

## Results
- `checker_2x2_global`: best acc=1.0, best mse=0.956093, graph=ext/IsingLearning/experiments/local_checkerboard_xor/runs/stable_local_langevin_stronger_20260514/checker_2x2_global/checker_2x2_global_best_graph.jld2
- `checker_4x4_global`: best acc=0.75, best mse=0.917103, graph=ext/IsingLearning/experiments/local_checkerboard_xor/runs/stable_local_langevin_stronger_20260514/checker_4x4_global/checker_4x4_global_best_graph.jld2

## Files
- Metrics CSV: `local_checkerboard_xor_metrics.csv`
- Progress PNG: `local_checkerboard_xor_progress.png`
- Per-config folders contain parameter SVGs and best graph JLD2 files.
