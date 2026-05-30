# R8 No-Output-Bias Relaxation/Beta Grid

- driver: `mnist_local_manager_grid.jl`
- radius: `8`
- epochs: `30`
- workers: `32`
- batchsize: `32`
- train/test per class: `100` / `20`
- output bias training: `false`
- relaxation sweeps: `25`, `50`, `100`, `150`
- beta values: `2.5`, `5.0`, `10.0`
- all runs start from random initialization
- checkpoints include sparse `J`, bias vector, optimizer state, update index, config, and source RNG
- launcher and launcher logs are kept in `launch/`

This grid tests whether the previous r8 collapse was caused by the one-vs-rest output-bias drift under `target_on=1`, `target_off=-1`.
