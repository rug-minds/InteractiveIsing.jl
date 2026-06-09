# Stochastic EP large-beta grid

Run date: 2026-06-08

This grid tested whether the locally connected Ising MNIST replication needs a finite, basin-moving nudge rather than the low-beta / infinitesimal regime.

## Shared setup

- Script: `diagnostics/lcl_stochastic_ep_training.jl`
- Launcher: `launch_lcl_stochastic_ep_large_beta_grid.ps1`
- Architecture: locally connected 6x6-window replication
- Workers: 32
- Batch size: 200
- Train subset: 500 examples per class
- Test subset: 100 examples per class
- Epochs: 80
- Relaxation sweeps: 25
- Temperature: 0.001
- Initial weight scale: 0.005
- Tangent nudging: enabled
- Stochastic nudged samples: `K = 8`

The stochastic estimator used one free trajectory and `K = 8` independent nudged trajectories per example. The nudged contrastive observables were averaged before applying the update. The free phase was not averaged over multiple independent initializations in this grid.

## Results

| run | beta | learning rate | best test | best epoch | final test |
| --- | ---: | ---: | ---: | ---: | ---: |
| `b3p0_lr0p0003_T0p001_s25_k8_e80` | 3.0 | 0.0003 | 0.613 | 75 | 0.585 |
| `b10p0_lr0p001_T0p001_s25_k8_e80` | 10.0 | 0.001 | 0.689 | 75 | 0.639 |
| `b30p0_lr0p001_T0p001_s25_k8_e80` | 30.0 | 0.001 | 0.690 | 30 | 0.660 |
| `b100p0_lr0p003_T0p001_s25_k8_e80` | 100.0 | 0.003 | 0.706 | 55 | 0.700 |

## Takeaways

- Larger finite beta clearly helped. Previous low/medium beta stochastic runs were essentially non-learning, while beta 3/10/30/100 all learned.
- Beta 100 was the best in this grid, reaching 70.6% test accuracy and ending at 70.0%.
- None of these large-beta runs showed the hard one-class collapse pattern; prediction counts stayed distributed across classes.
- The behavior supports the working hypothesis that scalar Ising states do not provide a useful local derivative under tiny nudges. A larger beta may be needed to push the system into a nearby target-improving basin so that the contrastive direction becomes informative.
- Beta 100 also showed drift after its best epoch, with weight/input norms continuing to grow. Follow-up runs should test lower learning rates, weight decay, beta schedules, and possibly longer training from the best checkpoint.

## Suggested next grid

- Explore beta in the range 50-200.
- Try lower learning rates around beta 100: `0.0003`, `0.0005`, `0.001`, `0.002`.
- Compare `K = 4`, `K = 8`, and `K = 16` at beta 100 to separate estimator noise from step-size effects.
- Test explicit regularization/weight decay or norm clipping, since beta 100 kept improving early while norms grew steadily.
