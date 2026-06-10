# Gradient Quality Diagnostic

This diagnostic compares two raw per-sample gradient estimates before minibatch scaling:

- forward: `plus_state - free_equilibrium_state`
- symmetric: `plus_state - minus_state`

The accepted update is the symmetric gradient. A sample gradient is discarded when the cosine overlap between the forward and symmetric buffers is below `ISING_MNIST_IF_GRADIENT_QUALITY_MIN_COSINE`.

## Implementation

File:

`mnist_784_120_40_softplus_margin_gradient_quality_adam.jl`

The implementation keeps the normal phase orchestration in `@Routine` and `@CompositeAlgorithm`. The added `AccumulateInputFieldGradientQualityRef!` process only computes two sample-gradient buffers, records their cosine/angle, and conditionally adds the symmetric buffer into the worker's accepted-gradient buffer.

CSV columns added:

- `gradient_quality_total`
- `gradient_quality_accepted`
- `gradient_quality_rejected`
- `gradient_quality_reject_fraction`
- `gradient_quality_avg_cosine`
- `gradient_quality_avg_angle_degrees`

Rejected samples are discarded from the estimator denominator. If a minibatch accepts no sample gradients, the Adam update is skipped.

## Smoke Run

Run folder:

`experiments/current/smoke_gradient_quality_r3`

Settings:

- `sweeps=0.01`
- `beta=0.025`
- `stepsize=0.35`
- `min_cosine=0.5`
- `workers=4`

Result:

- accepted `7 / 10`
- rejected `3 / 10`
- average cosine `0.6535`
- average angle `46.55 deg`

This verified that the counters and accepted-gradient path work.

## Real-Relaxation Diagnostic

Run folder:

`experiments/current/realrelax_fromscratch_b0p025_step0p35_cos0p5_s500`

Settings:

- from scratch
- `sweeps=500`
- `beta=0.025`
- `temp=0.00005`
- nudged temperature schedule `reverse_anneal`
- nudged temperature peak `0.00010`
- `stepsize=0.35`
- `lr=0.00010`
- `weight_decay=0.002`
- target weights `9.0 / 0.50`
- `min_cosine=0.5`
- `workers=32`
- training subset `13` per class, batch size `128`
- test subset `20` per class

Result:

- accepted `36 / 130`
- rejected `94 / 130`
- reject fraction `0.7231`
- average cosine `0.2383`
- average angle `75.07 deg`
- epoch test accuracy moved from `0.07` to `0.08` on the small test subset

Read: at random initialization and realistic relaxation, the forward and symmetric estimates disagree strongly under this setting. The threshold `0.5` is quite aggressive here; it discards nearly three quarters of samples.

## Sweep Count Diagnostic

Run folder:

`experiments/current/sweep_acceptance_20260610`

Shared settings:

- from scratch
- `beta=0.025`
- `temp=0.00005`
- nudged temperature schedule `reverse_anneal`
- nudged temperature peak `0.00010`
- `stepsize=0.35`
- `lr=0.00010`
- `weight_decay=0.002`
- target weights `9.0 / 0.50`
- `min_cosine=0.5`
- `workers=32`
- training subset `26` per class, batch size `128`
- test subset `20` per class

Results:

| sweeps | accepted | rejected | reject fraction | avg cosine | avg angle | test accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 64 / 260 | 196 / 260 | 0.7538 | 0.2351 | 75.34 deg | 0.115 |
| 1000 | 85 / 260 | 175 / 260 | 0.6731 | 0.2816 | 71.50 deg | 0.070 |
| 1500 | 88 / 260 | 172 / 260 | 0.6615 | 0.2751 | 71.49 deg | 0.065 |

Read: increasing sweeps from `500` to `1000` or `1500` improves acceptance from roughly `25%` to `34%`, but the average angle remains around `71 deg`. That is still far from the expected few-degree agreement. Under this setting, additional relaxation alone does not make the forward and symmetric gradient estimates align.

## Beta Diagnostic

Run folder:

`experiments/current/beta_acceptance_20260610_r2`

Shared settings:

- from scratch
- `sweeps=1000`
- `temp=0.00005`
- nudged temperature schedule `reverse_anneal`
- nudged temperature peak `0.00010`
- `stepsize=0.35`
- `lr=0.00010`
- `weight_decay=0.002`
- target weights `9.0 / 0.50`
- `min_cosine=0.5`
- `workers=32`
- training subset `26` per class, batch size `128`
- test subset `20` per class

Results:

| beta | accepted | rejected | reject fraction | avg cosine | avg angle | test accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.005 | 84 / 260 | 176 / 260 | 0.6769 | 0.2521 | 73.40 deg | 0.065 |
| 0.010 | 66 / 260 | 194 / 260 | 0.7462 | 0.1878 | 77.65 deg | 0.035 |
| 0.025 | 81 / 260 | 179 / 260 | 0.6885 | 0.2515 | 73.34 deg | 0.095 |
| 0.050 | 75 / 260 | 185 / 260 | 0.7115 | 0.2779 | 72.25 deg | 0.090 |
| 0.100 | 65 / 260 | 195 / 260 | 0.7500 | 0.2692 | 73.33 deg | 0.105 |

Read: varying beta by `20x` around `0.025` does not repair the estimator agreement. The best average angle in this sweep is still about `72 deg`, so the disagreement is not mainly explained by this beta range.

## Direct Symmetric Gradient Accumulation Check

Run folder:

`experiments/current/direct_contrast_check_20260610/beta0p025_s1000_step0p35_cos0p5`

Change tested:

- keep the existing `contrastive_gradient` behavior unchanged for old callers
- add opt-in `contrastive_gradient_new`
- use direct product-difference accumulation for the symmetric diagnostic buffer:
  - `dJ += -0.5 * (p_i * p_j - m_i * m_j)`
  - `db += -c * (p_i - m_i)`

Settings:

- from scratch
- `beta=0.025`
- `sweeps=1000`
- `temp=0.00005`
- nudged temperature schedule `reverse_anneal`
- nudged temperature peak `0.00010`
- `stepsize=0.35`
- `min_cosine=0.5`
- `workers=32`
- training subset `26` per class, batch size `128`

Result:

- accepted `89 / 260`
- rejected `171 / 260`
- reject fraction `0.6577`
- average cosine `0.2582`
- average angle `72.94 deg`
- test accuracy `0.095` on the small test subset

Read: the direct numerical accumulation path is cleaner and should be safer for symmetric EP, but it does not materially repair the gradient-angle diagnostic under this setting. The large disagreement is therefore unlikely to be mainly caused by Float32 plus-pass/minus-pass cancellation in the recurrent gradient buffer.
