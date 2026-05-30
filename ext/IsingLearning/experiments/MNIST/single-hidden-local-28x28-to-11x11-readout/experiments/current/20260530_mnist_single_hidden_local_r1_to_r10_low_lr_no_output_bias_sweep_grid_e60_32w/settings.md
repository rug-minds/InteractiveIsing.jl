# R1-R10 Low-LR No-Output-Bias Sweep Grid

- driver: `mnist_local_manager_grid.jl`
- radii: `1:10`
- epochs: `60`
- workers: `32`
- batchsize: `32`
- train/test per class: `100` / `20`
- output bias training: `false`
- weight learning rates W0/W12/W2O: `0.0004`
- bias learning rate: `0.00004`
- beta: `5.0`
- relaxation sweeps: `25`, `50`, `100`
- target encoding: original `target_on=1.0`, `target_off=-1.0`
- all runs start from random initialization
- folder layout: `s<SWEEPS>/r<RADIUS>_e60_low_lr_nooutbias`
- checkpoints include sparse `J`, bias vector, optimizer state, update index, config, and source RNG
- launcher and launcher logs are kept in `launch/`

This grid follows the r8 collapse diagnostic. The high-lr r8 runs collapse quickly; the low-lr/no-output-bias smoke was less degenerate, so this grid tests whether the successful radius region reappears when the update scale is reduced.
