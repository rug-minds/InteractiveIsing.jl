# LCL 6x6 Tangent Follow-Up Grid Results

Run folder:
`C:\Users\fenje\dev\InteractiveIsing.jl\ext\IsingLearning\experiments\MNIST\locally-connected-6x6-window-replication\experiments\current\20260606_011635_lcl_6x6_tangent_followup_grid`

All runs used `julia -t 32`, `32` manager workers, `ChannelWorkers()`, `100` training samples per class, `40` test samples per class, tangent output nudging, and `output_replicas = 1`.

| run | best test acc | best epoch | final test acc | note |
| --- | ---: | ---: | ---: | --- |
| `b3_lr3e-4_T0.001_s25_resume_e320` | 0.6850 | 315 | 0.6700 | continued the previous 0.6425 run from epoch 161 |
| `b3_lr3e-4_T0.001_s50_e240` | 0.6850 | 210 | 0.6675 | 50 sweeps is robust, but not clearly above 25 sweeps at this LR |
| `b3_lr2e-4_T0.001_s50_e240` | 0.6925 | 220 | 0.6650 | best 50-sweep run; lower LR helped slightly |
| `b3_lr5e-4_T0.001_s25_e220` | 0.7050 | 220 | 0.7050 | best run so far; still climbing at the end |
| `b3_lr3e-4_T0.0003_s25_e220` | 0.6525 | 210 | 0.6200 | colder than T=0.001 is worse |
| `b3_lr3e-4_T0.003_s25_e220` | 0.6525 | 205 | 0.6000 | warmer than T=0.001 is worse |

Current read:

- `T = 0.001` is the best temperature among `0.0003`, `0.001`, and `0.003` in this diagnostic.
- The best diagnostic setting is now `beta = 3`, `lr = 5e-4`, `T = 0.001`, `25` sweeps.
- The best run ended at its best epoch, so the next action is to continue it with optimizer state intact rather than restart it.
- More sweeps can help at lower LR, but the fastest-improving setting so far is the 25-sweep `lr = 5e-4` run.
