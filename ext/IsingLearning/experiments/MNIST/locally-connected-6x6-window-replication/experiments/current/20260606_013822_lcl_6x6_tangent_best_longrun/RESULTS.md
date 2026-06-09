# LCL 6x6 Tangent Best Long-Run Results

Run folder:
`C:\Users\fenje\dev\InteractiveIsing.jl\ext\IsingLearning\experiments\MNIST\locally-connected-6x6-window-replication\experiments\current\20260606_013822_lcl_6x6_tangent_best_longrun`

All runs used `julia -t 32`, `32` manager workers, `ChannelWorkers()`, `beta = 3`, `T = 0.001`, tangent output nudging, and `output_replicas = 1`.

| run | train/test per class | best test acc | best epoch | final test acc | epoch sec | note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `lr5e-4_s25_train100_resume_e500` | 100 / 40 | 0.7175 | 260 | 0.6950 | 0.40 | continued the previous 70.5% run; peaked early, then drifted down |
| `lr7e-4_s25_train100_e320` | 100 / 40 | 0.7075 | 110 | 0.7000 | 0.42 | faster LR did not beat 5e-4 |
| `lr5e-4_s50_train100_e320` | 100 / 40 | 0.7175 | 295 | 0.6925 | 0.60 | tied best but costs more per epoch |
| `lr5e-4_s25_train500_e120` | 500 / 100 | 0.6620 | 110 | 0.6600 | 1.51 | larger split learns; not a tiny-split artifact |

Current read:

- Best small-split diagnostic accuracy is `71.75%`.
- For these settings, `25` sweeps is more efficient than `50` sweeps. The 50-sweep run tied best accuracy but did not improve it.
- `lr = 5e-4` is better than `7e-4` on the small split.
- The 500/class split remains stable and learns to `66.2%` by 120 epochs, so the behavior transfers beyond the tiny diagnostic split.
- Next step is a larger balanced/full-candidate run with `beta = 3`, `lr = 5e-4`, `T = 0.001`, and `25` sweeps.
