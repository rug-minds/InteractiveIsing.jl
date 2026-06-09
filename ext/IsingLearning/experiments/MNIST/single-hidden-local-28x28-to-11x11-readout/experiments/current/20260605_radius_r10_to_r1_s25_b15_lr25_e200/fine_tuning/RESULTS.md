# Fine Tuning Results

Completed on 2026-06-05.

All runs started from the parent grid best checkpoints:

- r5 parent best: `0.500` at epoch `138`
- r10 parent best: `0.495` at epoch `151`

| branch | radius | sweeps | beta | lr | best test | best epoch | final test |
|---|---:|---:|---:|---:|---:|---:|---:|
| same_sampler_lr10 | 5 | 25/25 | 1.5 | 1e-5 | 0.505 | 174 | 0.350 |
| same_sampler_lr10 | 10 | 25/25 | 1.5 | 1e-5 | 0.530 | 231 | 0.440 |
| beta10_lr10 | 5 | 25/25 | 1.0 | 1e-5 | 0.490 | 152 | 0.420 |
| beta10_lr10 | 10 | 25/25 | 1.0 | 1e-5 | 0.525 | 181 | 0.410 |
| s35_beta10_lr10 | 5 | 35/35 | 1.0 | 1e-5 | 0.475 | 160 | 0.425 |
| s35_beta10_lr10 | 10 | 35/35 | 1.0 | 1e-5 | 0.535 | 174 | 0.350 |

Takeaways:

- r10 can be pushed above its parent-grid best. The best fine-tuned r10 result
  is `0.535`, from `35/35` sweeps, `beta=1.0`, `lr=1e-5`.
- r5 does not improve meaningfully from its parent best. The only branch above
  the parent result is `same_sampler_lr10`, and only by `0.005`.
- Fine tuning still decays after reaching the local best. The best checkpoints
  are the useful artifacts; final checkpoints are worse in every branch.
- The fact that r8 remained stronger in the parent grid is not explained by
  simply giving r5 or r10 slightly better continuation hyperparameters. r10 can
  move upward, but not beyond r8's parent-grid best of `0.625`.
