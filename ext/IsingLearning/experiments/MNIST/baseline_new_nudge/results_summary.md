# Softplus-Margin Nudge MNIST Results

Last updated: 2026-06-09

## Setup

This folder tests the `784 -> 120 -> 40` MNIST baseline with the original
one-sided free/nudged EqProp learning rule, but replaces the quadratic output
clamping term with `InteractiveIsing.SoftplusMarginNudging`.

Common settings for the scout runs:

- Julia threads/workers: `32 / 32`
- Architecture: `784 -> 120 -> 40`
- Batch size: `128`
- Training subset: `1280` samples per class, exactly `100` full batches per epoch
- Test subset: `100` samples per class
- Train-eval subset: `100` samples per class
- Sweeps: `500`
- Optimizer: `Optimisers.Adam`
- Evaluation: every epoch
- Early stopping: stop after two consecutive evaluated test-accuracy drops
- Sparse recurrent weights: symmetrized at initialization, after gradient accumulation, and after Adam updates

All recorded `symmetry_error` and `grad_symmetry_error` values stayed at `0.0`.

## Main Takeaways

The new softplus-margin nudge can learn at much lower beta than the old
quadratic-clamping baseline. The useful region so far is around:

```text
beta = 0.1
temperature = 5e-5
learning rate = 0.0015
weight decay = 0.001
```

The best completed scout result so far is:

```text
beta = 0.1
temperature = 5e-5
weight_decay = 0.001
best test accuracy = 0.558 at epoch 2
final recorded test accuracy = 0.521 at epoch 4
```

The dominant failure mode is still post-peak collapse: recurrent and especially
input-projection norms keep growing, after which prediction counts concentrate
into a few classes and accuracy falls.

Increasing relaxation from `500` to `1000` sweeps did not improve the best
accuracy in the first diagnostic. It reduced the best observed loss, but the
classifier still concentrated into a few classes and stayed below the best
`500`-sweep fixed-decay run.

## Low-Beta, Low-Temperature Scout

Grid root:

```text
experiments/current/scout_grid_lowbeta_lowtemp_wd1e4_20260609
```

Fixed settings:

```text
temperature = 5e-5
weight_decay = 1e-4
learning rate = 0.0015
```

| beta | best epoch | best test accuracy | best test loss | final epoch | final test accuracy | final test loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 2 | 0.405 | 15.31118 | 10 | 0.361 | 15.97124 |
| 0.10 | 4 | 0.513 | 12.955752 | 6 | 0.498 | 14.235217 |
| 0.20 | 2 | 0.483 | 26.30074 | 4 | 0.410 | 30.53182 |

Interpretation:

- `beta = 0.1` was clearly the best of this first low-beta pass.
- `beta = 0.05` learns but appears under-driven.
- `beta = 0.2` gives a stronger early response but worse loss and faster collapse.

## Weight-Decay Focus Grid

Grid root:

```text
experiments/current/focus_grid_weight_decay_20260609
```

Fixed settings:

```text
temperature = 5e-5
learning rate = 0.0015
```

Completed fixed-decay runs:

| beta | weight decay | best epoch | best test accuracy | best test loss | final epoch | final test accuracy | final test loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.10 | 0.0003 | 3 | 0.478 | 16.705326 | 7 | 0.401 | 22.079105 |
| 0.10 | 0.0010 | 2 | 0.558 | 16.317978 | 4 | 0.521 | 18.097399 |
| 0.10 | 0.0030 | 1 | 0.476 | 19.638191 | 3 | 0.456 | 19.367655 |
| 0.20 | 0.0003 | 3 | 0.460 | 23.45958 | 5 | 0.401 | 28.144684 |
| 0.20 | 0.0010 | 1 | 0.431 | 28.762209 | 6 | 0.270 | 43.76586 |
| 0.20 | 0.0030 | 1 | 0.428 | 30.10329 | 7 | 0.185 | 46.85131 |

Interpretation:

- Increasing weight decay helped at `beta = 0.1` up to `0.001`.
- Too much fixed decay (`0.003`) suppressed the useful signal and did not prevent later degradation.
- `beta = 0.2` was consistently worse in the fixed-decay focus grid.

## Adaptive Weight Decay

Implemented controller:

```text
effective_decay = base_decay + gain * max(0, current_norm / target_norm - 1)
```

with separate effective decay for recurrent weights and input-projection
weights, capped by `adaptive_max_decay`.

First adaptive run:

```text
beta = 0.1
temperature = 5e-5
base weight_decay = 1e-4
adaptive_w_norm = 1.8
adaptive_w_input_norm = 5.0
adaptive_gain = 7.5e-4
adaptive_max_decay = 0.003
```

Result:

| beta | adaptive | best epoch | best test accuracy | best test loss | final epoch | final test accuracy | final test loss |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.10 | true | 1 | 0.466 | 18.68113 | 3 | 0.266 | 30.325863 |

This configuration did not help. The input-projection target was too loose:
`w_input_norm` stayed below `5.0` until collapse was already underway, so the
input effective decay stayed at the base `1e-4`.

Current in-progress adaptive run:

```text
beta = 0.2
temperature = 5e-5
base weight_decay = 1e-4
adaptive = true
```

Latest recorded values:

| epoch | train accuracy | test accuracy | test loss | w norm | w input norm | effective w decay | effective w input decay |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.074 | 0.074 | 74.010315 | 0.48583895 | 1.5314544 | 0.0001 | 0.0001 |
| 1 | 0.443 | 0.476 | 28.550774 | 2.0537286 | 3.8041975 | 0.00020572025 | 0.0001 |
| 2 | 0.488 | 0.508 | 26.561821 | 2.4931147 | 4.679713 | 0.00038879784 | 0.0001 |

This run is not complete at the time of this note. It is already better than
the fixed-decay `beta = 0.2` runs, but still has high loss compared with the
best `beta = 0.1, weight_decay = 0.001` run.

## Recommended Next Tests

Use `beta = 0.1`, `temperature = 5e-5`, `learning_rate = 0.0015` as the anchor.

Recommended fixed-decay refinement:

```text
weight_decay = 0.0007, 0.0010, 0.0015
```

Recommended adaptive refinement:

```text
base weight_decay = 0.0003 or 0.0005
adaptive_w_norm = 1.8
adaptive_w_input_norm = 3.5 or 4.0
adaptive_decay_gain = 0.001
adaptive_max_decay = 0.003
```

The input-projection norm should be controlled more aggressively than in the
first adaptive run. The previous target `5.0` was too high for early collapse
prevention.

## Equilibrium and Langevin Stepsize Diagnostic

Grid root:

```text
experiments/current/equilibrium_stepsize_diagnostic_20260609
```

Fixed settings:

```text
beta = 0.1
temperature = 5e-5
learning rate = 0.0015
weight_decay = 0.001
train/test per class = 1280 / 100
```

| sweeps | stepsize | status | best epoch | best test accuracy | best test loss | final epoch | final test accuracy | final test loss |
| ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 0.75 | done | 8 | 0.489 | 13.743121 | 10 | 0.357 | 19.261122 |
| 1000 | 1.00 | stopped after epoch 4 | 1 | 0.479 | 18.927998 | 4 | 0.455 | 19.549547 |

Interpretation:

- `1000` sweeps with stepsize `0.75` was stable but did not beat the `500`-sweep
  anchor (`0.558` best test accuracy).
- The best loss for `1000` sweeps / stepsize `0.75` was lower than the anchor's
  best-loss row, but accuracy stayed worse, so better relaxation alone is not
  fixing the class-collapse issue.
- Stepsize `1.0` learned quickly in epoch 1 but produced larger gradients and
  worse loss, then stayed below `0.48`; this looks too aggressive at the current
  learning rate and decay.
- The planned `1500`-sweep case was skipped after these two diagnostics because
  the extra relaxation cost was not buying accuracy.
