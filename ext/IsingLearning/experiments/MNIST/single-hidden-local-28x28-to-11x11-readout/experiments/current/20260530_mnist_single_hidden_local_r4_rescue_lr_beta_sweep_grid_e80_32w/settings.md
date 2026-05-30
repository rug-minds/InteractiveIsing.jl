# R4 Rescue Grid

- driver: `mnist_local_manager_grid.jl`
- radius: `4`
- epochs: `80`
- workers: `32`
- batchsize: `32`
- train/test per class: `100` / `20`
- output bias training: `false`
- target encoding: original `target_on=1.0`, `target_off=-1.0`
- weight learning rates W0/W12/W2O: `0.0002`, `0.0004`, `0.0008`
- bias learning rates: `0.00002`, `0.00004`, `0.00008` matched to the weight scale
- beta values: `1.0`, `2.5`, `5.0`
- relaxation sweeps: `50`, `100`, `200`
- all runs start from random initialization
- folder layout: `lr<LR>/s<SWEEPS>/beta<BETA>/r4_e80`
- checkpoints include sparse `J`, bias vector, optimizer state, update index, config, and source RNG
- launcher and launcher logs are kept in `launch/`

This targets the observed radius crossing from the earlier grid: `r=3` showed weak learning (`0.24`/`0.28` best), while `r=4` was the closest larger radius that did not learn. The search uses lower update scales and no output-bias training to avoid the one-class collapse seen in the r8 diagnostics.
