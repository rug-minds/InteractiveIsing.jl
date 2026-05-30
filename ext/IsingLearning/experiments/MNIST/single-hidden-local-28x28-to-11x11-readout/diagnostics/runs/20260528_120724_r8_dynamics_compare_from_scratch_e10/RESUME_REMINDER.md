# Resume Reminder

This diagnostic was stopped because the dynamics comparison only needs 5 epochs, not 10.

Resume later with a new 5-epoch from-scratch comparison for the single-hidden-local MNIST model:

- architecture: `784 -> 784 -> 121 -> 40`, local radius `r=8`
- data: `100` train and `20` test samples per class
- batch size/workers: `32` / `32`
- relaxation: `50` free sweeps, `50` nudged sweeps, reads `3/3`
- optimizer/LR: Adam, W0/W12/W2O `0.004`, B `0.0004`, gradient normalization `mean`
- initialization: scratch only; no checkpoint
- dynamics to compare: Metropolis, unadjusted local Langevin, unadjusted global Langevin
- Langevin stepsizes: `0.03`, `0.10`, `0.30`

The aborted 10-epoch launcher only completed partial Metropolis metrics. Ignore it for the final comparison.

Use the updated `mnist_local_manager_grid.jl` for the next run. It includes `ProgressMeter.jl`, `SparseMatrixCSC` adjacency storage, shared source/worker J and base bias, and a sparse-layout gradient path. The old implementation was copied to `mnist_local_manager_grid_OLD.jl`.
