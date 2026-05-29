# Output-State Time-Averaged XOR

This folder tests the correction from the earlier time-average experiment:
average the **state**, then compute the usual EqProp contrastive gradient.

The previous `simple_timeavg_xor` folder averaged the instantaneous parameter
derivative over a nudged trajectory. That is a different estimator. It is not
the same as taking a plus/minus EqProp state and then applying the existing
`contrastive_gradient(graph, s_plus, s_minus, beta)` rule.

## What Is Averaged

There are two different averages in the code:

1. Output averaging for validation.
   The output spin is averaged over time after a burn-in. This only changes the
   readout used to compute MSE and accuracy. It is valid for evaluation, but by
   itself it cannot update `J`, because the `J` gradient needs products of all
   spin states, not only the output coordinate.

2. Whole-state averaging for training.
   During the plus and minus nudged phases, the experiment samples the complete
   spin vector after full sweeps. It then computes
   `contrastive_gradient(graph, mean_plus_state, mean_minus_state, beta)`.
   This is the well-defined state-averaged EqProp variant tested here.

So: output averaging measures the final classifier; whole-state averaging is the
actual learning signal.

## Current Implementation

The main files are:

- `output_state_2_4_1_timeavg_learning.jl`
- `output_state_2_4_1_grid.jl`

The worker process runs:

1. free relaxation,
2. plus nudged relaxation,
3. full-state sampling over repeated full sweeps,
4. minus nudged relaxation,
5. full-state sampling over repeated full sweeps.

The manager consume step then computes the normal symmetric EqProp contrastive
gradient from the two averaged full states. The gradient is accumulated in the
worker buffer and flushed once per batch.

## Result From 2026-05-13

Run folder:

`ext/IsingLearning/experiments/output_state_timeavg_xor/runs/timeavg_grid_20260513_181757`

Configuration summary:

| config | whole-state samples | free/nudged sweeps | best MSE | best accuracy | best mean outputs |
|---|---:|---:|---:|---:|---|
| `f300_n300_avg1_lr002` | 1 | 300/300 | 0.529677 | 0.75 | `[-0.6289, -0.0035, 0.6937, -0.0619]` |
| `f300_n300_avg5_lr002` | 5 | 300/300 | 0.396540 | 0.75 | `[-0.5761, -0.0774, 0.6644, -0.6352]` |
| `f300_n300_avg20_lr002` | 20 | 300/300 | 0.291582 | 1.00 | `[-0.5763, 0.1372, 0.6593, -0.6446]` |
| `f600_n600_avg1_lr002` | 1 | 600/600 | 0.561615 | 0.75 | `[-0.5833, -0.0055, 0.6671, -0.0248]` |

Interpretation:

- Whole-state averaging helped in this run. `avg20` was the only tested setting
  that reached accuracy `1.0`.
- The output magnitudes are still too small for the target `[-1, +1, +1, -1]`,
  so the MSE remains much higher than the earlier successful endpoint-gradient
  scalar XOR runs.
- Increasing free/nudged sweeps from `300/300` to `600/600` did not help when
  only one plus/minus state sample was used.
- The useful comparison is not "output averaging versus whole-state averaging"
  as two equivalent training rules. Output averaging is a readout statistic;
  whole-state averaging is a complete state estimator that can drive the
  existing `J` and `b` gradients.

## Next Things To Try

- Run `avg20` longer than 1200 epochs.
- Try `avg20` with slightly lower temperature or larger output clamping to push
  the learned output means closer to `±1`.
- Compare against the known endpoint-gradient successful run in
  `scalar_2_4_1_xor_langevin.md`; the present state-averaged estimator is more
  stable in sign but not yet as sharp in MSE.

## Longer `avg20` Check

Run folder:

`ext/IsingLearning/experiments/output_state_timeavg_xor/runs/timeavg_grid_20260513_182332`

The same `f300_n300_avg20_lr002` setting was run to 3000 epochs. It reached:

| epoch | MSE | accuracy | mean outputs |
|---:|---:|---:|---|
| 1400 | 0.218819 | 1.00 | `[-0.566, 0.341, 0.665, -0.624]` |
| 3000 | 0.716838 | 0.75 | `[-0.209, -0.014, 0.187, -0.257]` |

So the longer run confirmed that whole-state averaging can get to a better
point than the 1200-epoch sweep, but it also drifts away afterward. For this
specific recipe the best graph should be restored/saved; continuing the same
optimizer settings does not monotonically improve the readout MSE.

## 8000-Epoch Learning-Rate Check

Run folder:

`ext/IsingLearning/experiments/output_state_timeavg_xor/runs/timeavg_grid_20260513_185605`

All three runs used `300/300` free/nudged sweeps and `20` whole-state samples.

| config | best epoch | best MSE | best accuracy | final MSE | final accuracy |
|---|---:|---:|---:|---:|---:|
| `lr=0.002` | 1500 | 0.177633 | 1.00 | 0.940449 | 0.75 |
| `lr=0.001` | 3000 | 0.132611 | 1.00 | 0.866755 | 1.00 |
| `lr=0.0005` | 6250 | 0.131991 | 1.00 | 0.178195 | 1.00 |

The lower learning rates make the good regime last longer. The best MSE is now
about `0.132`, reached by both `lr=0.001` and `lr=0.0005`. The `lr=0.0005`
run is also much more stable by epoch 8000 than the larger learning rates.

This still has the same qualitative issue: the signs classify XOR correctly,
but the output magnitudes settle around `0.55-0.73`, not close to `±1`. More
epochs alone are not enough; the next useful tuning axis is probably stronger
nudging or temperature/stepsize changes after the state-averaged estimator has
settled.

## Energy Scale And Temperature Check

Run folder:

`ext/IsingLearning/experiments/output_state_timeavg_xor/runs/timeavg_grid_20260513_223547`

These runs kept `300/300` free/nudged sweeps, `20` whole-state samples, and
`lr=0.0005`, then varied the effective energy/temperature knobs.

| config | change | best epoch | best MSE | best accuracy | best mean outputs |
|---|---|---:|---:|---:|---|
| `base_lr0005` | baseline | 5500 | 0.125654 | 1.00 | `[-0.5759, 0.7047, 0.6532, -0.6604]` |
| `temp0025_lr0005` | lower `T` from `0.005` to `0.0025` | 6000 | 0.124458 | 1.00 | `[-0.6270, 0.6754, 0.6873, -0.6056]` |
| `beta3_lr0005` | stronger clamp, `β=3` | 5250 | 0.797905 | 1.00 | `[-0.1341, 0.1095, 0.0189, -0.1716]` |
| `scale025_lr0005` | larger initial `J` scale, `0.25` | 5750 | 0.145137 | 1.00 | `[-0.5874, 0.6142, 0.6590, -0.6191]` |

Interpretation:

- Lowering temperature helped only marginally.
- Increasing `β` to `3` was harmful: the output means collapsed toward zero.
- Larger initial weight scale did not help. This does not rule out a learned or
  post-update global scale, but it says that simply starting bigger is not the
  missing ingredient for this recipe.
- The best state-averaged result so far is `0.124458`, still above the `0.1`
  target.

## About Learning A Global Weight Scale

It is possible to introduce a positive global scale `λ` multiplying all
trainable non-clamping energy terms, for example

`H(s; λ, θ) = λ H_model(s; θ) + H_clamp(s; β, y)`.

EqProp can produce a gradient for `λ` because

`∂H/∂λ = H_model(s; θ)`.

However, this parameter is not innocuous:

- Increasing `λ` lowers the effective temperature, since the Boltzmann weight is
  `exp(-λ H_model / T)`.
- If `β` is not scaled with `λ`, the nudging term becomes relatively weaker.
- Without regularization, a learned `λ` can become a noise-suppression knob
  instead of learning a better classifier.

A safer version is to keep `λ` outside the optimizer as a controlled schedule or
post-update normalization, then test whether restoring/evaluating with a larger
model-energy scale lowers the readout variance. If `λ` is learned, it should be
regularized and probably applied consistently to `β` or interpreted explicitly
as an adaptive effective-temperature parameter.

## Validation-Only Temperature Check

Run folder:

`ext/IsingLearning/experiments/output_state_timeavg_xor/runs/timeavg_grid_20260513_224406`

Training temperature was fixed at `T=0.005`. Only validation temperature changed.

| config | validation `T` | best epoch | best MSE | best accuracy | best mean outputs |
|---|---:|---:|---:|---:|---|
| `trainT005_evalT005` | 0.0050 | 5500 | 0.123487 | 1.00 | `[-0.5893, 0.6955, 0.6532, -0.6650]` |
| `trainT005_evalT0025` | 0.0025 | 6250 | 0.124179 | 1.00 | `[-0.6130, 0.7315, 0.6798, -0.5849]` |
| `trainT005_evalT001` | 0.0010 | 6750 | 0.132209 | 1.00 | `[-0.6277, 0.7353, 0.7215, -0.5075]` |
| `trainT005_evalT0005` | 0.0005 | 6000 | 0.115004 | 1.00 | `[-0.6205, 0.7486, 0.6863, -0.6070]` |

Interpretation:

- Colder validation helps a little, with the best result now `0.115004`.
- It does not collapse the MSE toward zero. The learned outputs are still not
  close enough to `±1`; validation noise is only part of the error.
- This supports the idea that effective energy scale matters, but the current
  graph parameters themselves still set output fixed points around
  `0.6-0.75`, not near the bounds.
