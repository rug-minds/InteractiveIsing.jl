# LCL 6x6 Tangent Focused Grid Results

Run folder:
`C:\Users\fenje\dev\InteractiveIsing.jl\ext\IsingLearning\experiments\MNIST\locally-connected-6x6-window-replication\experiments\current\20260606_005633_lcl_6x6_tangent_focused_grid`

All runs used `julia -t 32`, `32` manager workers, `ChannelWorkers()`, `100` training samples per class, `40` test samples per class, tangent output nudging, and temperature `T = 0.001` unless stated otherwise.

| run | best test acc | best epoch | final test acc | note |
| --- | ---: | ---: | ---: | --- |
| `b3_lr1e-4_s25_rep1_e200` | 0.4975 | 130 | 0.4400 | confirms the earlier resumed 0.4575 run was not an artifact |
| `b3_lr1e-4_s50_rep1_e200` | 0.6100 | 195 | 0.5825 | more sweeps helped strongly in this LCL/tangent setup |
| `b3_lr3e-4_s25_rep1_e160` | 0.6425 | 160 | 0.6425 | best focused result so far; still climbing at the end |
| `b5_lr3e-5_s25_rep1_e160` | 0.1750 | 160 | 0.1750 | too little learning |
| `b5_lr1e-4_s25_rep1_e160` | 0.5025 | 130 | 0.4600 | learns, but worse than beta 3 with higher LR |
| `b3_lr1e-4_s25_rep4_e200` | 0.4600 | 165 | 0.4225 | output replicas did not help this LCL topology/run |

Current read:

- The useful region is not simply "higher beta". `beta = 5` can learn, but did not beat `beta = 3`.
- More relaxation did not hurt here; `50` sweeps beat `25` sweeps at the same `beta = 3`, `lr = 1e-4`.
- The best run is `beta = 3`, `lr = 3e-4`, `25` sweeps, and it ended at its best epoch. It should be continued with optimizer state intact.
- A higher-sweep variant at `beta = 3`, `lr = 3e-4` is the next most important fresh run.
