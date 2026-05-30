# R8 Low-LR No-Output-Bias Relaxation/Beta Grid

- driver: `mnist_local_manager_grid.jl`
- radius: `8`
- epochs: `30`
- workers: `32`
- batchsize: `32`
- train/test per class: `100` / `20`
- output bias training: `false`
- weight learning rates W0/W12/W2O: `0.0004`
- bias learning rate: `0.00004`
- relaxation sweeps: `25`, `50`, `100`
- beta values: `2.5`, `5.0`, `10.0`
- target encoding: original `target_on=1.0`, `target_off=-1.0`
- all runs start from random initialization
- checkpoints include sparse `J`, bias vector, optimizer state, update index, config, and source RNG
- launcher and launcher logs are kept in `launch/`

This grid follows the r8 collapse diagnostic. The high-lr no-output-bias smoke still collapsed, while the low-lr smoke did not immediately collapse, so this grid tests whether r8 needs the lower update scale.
