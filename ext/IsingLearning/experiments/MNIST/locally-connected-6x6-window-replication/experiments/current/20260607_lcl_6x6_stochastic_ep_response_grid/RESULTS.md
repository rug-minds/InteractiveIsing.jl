# LCL stochastic EP response grid

Date: 2026-06-07

## Purpose

This diagnostic tests whether a stochastic-equilibrium-propagation style estimator improves the scalar Ising LCL model's error signal.

Instead of using one free state and one nudged state directly, the diagnostic runs multiple independent nudged stochastic rollouts from the same free equilibrium and averages the contrastive observables:

- graph weights use averaged differences of `s_i * s_j`;
- biases use averaged differences of `s_i`;
- input projection weights use averaged differences of `x_input * s_hidden`.

This is deliberately not product-of-averaged-states. The estimator averages the sufficient statistics used by the contrastive gradient.

## Setup

- Script: `diagnostics/lcl_stochastic_ep_training.jl`
- Launcher: `launch_lcl_stochastic_ep_diagnostic_grid.ps1`
- Julia threads: 32
- Manager workers: 32
- Training split: balanced 500 examples per class
- Test split: balanced 100 examples per class
- Epochs: 80
- Batch size: 200
- Sweeps: 25
- Temperature: `0.001`
- Stepsize: `0.5`
- Initial weight scale: `0.005`
- Tangent nudge: enabled
- Stochastic nudged samples: 4

## Results

| Run | Best test acc | Best epoch | Final test acc | Final prediction counts |
| --- | ---: | ---: | ---: | --- |
| `b0p1_lr0p0001_T0p001_s25_k4_e80` | 0.121 | 75 | 0.092 | `92-102-90-99-99-105-92-90-112-119` |
| `b0p3_lr0p0001_T0p001_s25_k4_e80` | 0.118 | 70 | 0.117 | `87-106-110-103-103-97-111-90-81-112` |
| `b0p3_lr0p0003_T0p001_s25_k4_e80` | 0.114 | 15 | 0.094 | `99-88-89-117-90-105-86-108-114-104` |
| `b1p0_lr0p0003_T0p001_s25_k4_e80` | 0.123 | 50 | 0.119 | `113-93-102-109-80-93-112-123-95-80` |

## Interpretation

The stochastic estimator is wired into the real manager training path: gradient norms are nonzero, checkpoints are written, and weights move.

The important result is negative but useful: these runs do not collapse to one class, but they also do not learn. Prediction counts stay roughly balanced around 100 per class, while accuracy stays near chance. This suggests the previous one-class collapse was not the only failure mode; removing collapse by averaging noisy stochastic nudged observables is not enough to produce a class-aligned error direction.

The runs show a common pattern:

- early gradients are large;
- weights and biases drift;
- gradient norms quickly shrink;
- test accuracy remains near chance.

So the current scalar Ising LCL model still appears to lack a smooth, informative differential response to nudging. Stochastic observable averaging reduces pathological collapse, but it does not yet recover the XY-paper-style propagation signal.

## Next diagnostics

The next useful diagnostic is not another long training grid. We should directly measure response alignment:

- start from a free equilibrium;
- apply positive and negative nudges;
- measure whether output and hidden observables move in the target-improving direction;
- sweep beta, temperature, sweeps, and weight scale;
- log `free -> nudged` observable deltas before any optimizer update.

If alignment is not present at the single-example/single-minibatch level, training cannot work regardless of optimizer settings.
