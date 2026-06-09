# Surrogate Attractor MNIST Experiment Summary

Date: 2026-06-09

This folder tests the final-minimum reward rule for the MNIST `784-120-40`
reduced Ising classifier. The goal was to see whether rewarding only the final
sampled minima gives a useful learning signal before moving to trajectory-level
REINFORCE rules.

## Implemented Rule

For each input, the dynamics relaxes to final attractor samples `m`. Each final
minimum receives a scalar reward `R_y(m)` from the output readout. The update is
the batch-baseline final-minimum covariance rule:

```math
\Delta J_{ij}
\propto
\langle R m_i m_j\rangle
-
\langle R\rangle\langle m_i m_j\rangle
```

```math
\Delta h_i
\propto
\langle R m_i\rangle
-
\langle R\rangle\langle m_i\rangle .
```

The current implementation applies the same rule to:

- sparse graph couplings,
- graph bias fields,
- dense input-to-hidden field projection weights.

The target label only enters through the reward. There is no nudging term and no
transition-probability eligibility trace in the active rule.

## Implementation Notes

- Uses `ProcessManager` with `ChannelWorkers()` for both training and validation.
- Workers are persistent and keep preallocated local buffers through the run.
- Validation workers are also preallocated and reused.
- Attractor statistics are accumulated online in worker-local buffers.
- Coupling accumulation loops directly over CSC nonzero storage slots, not over
  all spin pairs.
- Dense input-field weights loop over their dense parameter storage.
- The batch-average reward baseline is subtracted by the manager before Adam.
- A sweep means one full pass worth of single-spin `LocalLangevin` updates over
  the active hidden/output spins. For this model, active spins are `160`, so
  `25` sweeps means `4000` single-spin updates.
- The active implementation does not use a separate surrogate temperature in the
  gradient. Temperature only affects the dynamics/noise.

## Sampling Protocol

For each training example:

1. Initialize and apply the projected input field.
2. Relax for `sweeps * active_units` Langevin steps.
3. Repeat `covariance_samples` times:
   - inject Gaussian state noise,
   - run a short high-step-size kick,
   - relax for `covariance_sample_sweeps * active_units` Langevin steps,
   - record only the final state as a sampled attractor,
   - attach reward to that final state.

Most runs used:

- `covariance_samples = 20`,
- `batchsize = 128`,
- `workers = 32`,
- 100 balanced train samples per class,
- 100 balanced test samples per class.

Each full minibatch therefore accumulates up to `128 * 20 = 2560` final-minimum
samples. The last minibatch of the 1000-sample subset has fewer examples and
accumulates `2080` samples.

## Timing Learned

With `25` initial sweeps and `25` sample sweeps:

- first minibatch in a fresh Julia/session: about `18.3s`, mostly compilation and
  first-call setup,
- second minibatch in the same session: about `0.18s`,
- 1000-sample epoch after warmup: about `1.4s` to `1.6s`.

With `50` sweeps and `50` sample sweeps:

- first epoch includes setup/warmup and was around `9s` to `10s`,
- later 1000-sample epochs were about `2.8s` to `3.8s`.

This suggests the ProcessManager/ChannelWorkers setup is not the main bottleneck
after warmup. The expensive part is the first compilation/setup pass and then the
expected relaxation/sample work.

## Results So Far

All results below are on the 1000-train / 1000-test balanced subset unless noted.
They are not full-MNIST results.

### Earlier 10-Epoch Runs at `temp = 0.01`, `lr = 0.003`

| sweeps | sample sweeps | best test acc | final test acc | final loss |
|---:|---:|---:|---:|---:|
| 25 | 25 | 0.192 | 0.167 | 73.13 |
| 50 | 50 | 0.192 | 0.166 | 72.24 |

These runs showed a real above-chance signal, but accuracy plateaued around
`19%` and losses rose substantially.

### Low-Temperature 10-Epoch Grid

| temp | lr | sweeps | sample sweeps | best test acc | final test acc | final loss |
|---:|---:|---:|---:|---:|---:|---:|
| 0.001 | 0.001 | 25 | 25 | 0.232 | 0.199 | 68.35 |
| 0.001 | 0.001 | 50 | 50 | 0.253 | 0.228 | 69.22 |
| 0.001 | 0.0005 | 25 | 25 | 0.173 | 0.154 | 59.12 |
| 0.001 | 0.0005 | 50 | 50 | 0.199 | 0.183 | 66.89 |
| 0.003 | 0.001 | 25 | 25 | 0.192 | 0.192 | 67.68 |
| 0.003 | 0.001 | 50 | 50 | 0.203 | 0.182 | 72.33 |
| 0.003 | 0.0005 | 25 | 25 | 0.141 | 0.138 | 55.90 |
| 0.003 | 0.0005 | 50 | 50 | 0.187 | 0.180 | 54.22 |

Best observed setting so far:

```text
temp = 0.001
lr = 0.001
sweeps = 50
covariance_sample_sweeps = 50
best test accuracy = 0.253
final test accuracy = 0.228
```

### Super-Low-Temperature Deeper-Relaxation Runs

The script validator currently requires positive `temp`, so exact zero was not
run here. Instead, these runs used `temp = 1e-6`, which is effectively
zero-temperature dynamics while keeping the existing experiment config unchanged.

| temp | lr | sweeps | sample sweeps | best test acc | final test acc | final loss |
|---:|---:|---:|---:|---:|---:|---:|
| 1e-6 | 0.001 | 100 | 100 | 0.207 | 0.194 | 71.45 |
| 1e-6 | 0.001 | 200 | 200 | 0.297 | 0.297 | 71.01 |

The `200/200` run is the best observed accuracy so far:

```text
temp = 1e-6
lr = 0.001
sweeps = 200
covariance_sample_sweeps = 200
best test accuracy = 0.297
final test accuracy = 0.297
```

This supports the idea that, for this final-minimum rule, deeper relaxation can
matter more than moderate thermal sampling. The `100/100` setting was worse than
the earlier `0.001`, `50/50` run, so the effect is not simply "lower temperature
is always better"; enough relaxation at the very low temperature appears
important.

## What We Learned

The final-minimum reward rule does produce nonzero gradients and learns above
chance. This is important: the reward-only-at-minima rule is not dead on arrival.

Lower physical temperature helped when paired with enough relaxation. The best
low-temperature run reached `29.7%` test accuracy on the balanced 1000-example
test subset, compared with about `19%` in the earlier `temp = 0.01`, `lr = 0.003`
runs.

More sweeps helped at `temp = 0.001`, `lr = 0.001`: `50/50` outperformed `25/25`.
At `temp = 1e-6`, `200/200` substantially outperformed `100/100`. More sweeps
are still not universally better across all settings, but the current evidence
favors trying deeper relaxation for near-zero-temperature minima sampling.

The rule tends to increase parameter norms and can concentrate predictions into
a subset of classes. This is visible in rising losses and imbalanced prediction
counts even when accuracy improves. A better run should track both accuracy and
prediction distribution, not only best accuracy.

The lower learning rate `0.0005` reduced loss growth in some cases but also
weakened the accuracy gain. The current best tradeoff is still `lr = 0.001`.

## Open Questions

- Does the rule continue improving on a longer run from the best setting, or does
  it saturate around `25%`?
- Would weight decay or gradient clipping prevent prediction concentration while
  preserving the early accuracy gains?
- Is `logprob` reward the right scalar reward, or would `margin` give cleaner
  gradients for this attractor-ranking rule?
- Are the sampled states actually stable minima, or just relaxed low-temperature
  states? We have not yet added a local stability diagnostic.
- Does lowering attractor energy increase reach frequency for correct minima, or
  is the rule mostly changing readout bias/field geometry?

## Suggested Next Runs

The next most useful run is a longer continuation of the best setting:

```text
temp = 1e-6
lr = 0.001
sweeps = 200
covariance_sample_sweeps = 200
epochs = 30 or 50
```

Track:

- test accuracy,
- test loss,
- prediction counts,
- parameter norms,
- reward mean/std,
- local stability margins if implemented.

A second useful grid is:

```text
temp = 0.001
sweeps = 50
covariance_sample_sweeps = 50
lr in {0.001, 0.0007, 0.0005}
weight_decay in {0, 1e-4, 1e-3}
```

The purpose would be to test whether regularization keeps the useful early
learning signal while reducing class concentration.

## 2026-06-09 Progress-Printed Loose-Relaxation Run

I added `ProgressMeter` reporting to
`mnist_784_120_40_surrogate_attractor_adam.jl`. The training loop now reports
each minibatch with the current sample count, batch reward mean/std, and number
of sampled minima. Plain `println` batch lines are still emitted, so the log
remains readable even when stderr progress control codes are not rendered by a
terminal.

Run directory:

```text
experiments/current/real_loose_progress_200pc_30ep
```

Settings:

```text
workers = 32
train/test/train-eval per class = 200 / 200 / 200
epochs = 30
batchsize = 128
temp = 1e-4
lr = 0.001
sweeps = 25
covariance_samples = 20
covariance_sample_sweeps = 50
covariance_kick_steps = 25
covariance_noise_temp_factor = 4
reward = logprob
```

This was not a true dynamic `until equilibrium` training run. It used the
standard fixed-step manager routine, with settings chosen from the looser
equilibrium/resampling diagnostics: a 25-sweep initial relaxation and 50 sweeps
after each noise kick.

Timing:

```text
epoch 1: 17.19s including first epoch overhead
epochs 2-30: mostly 5.5s to 7.2s per epoch
```

Evaluation checkpoints:

```text
epoch 0:  train 0.0830, test 0.0965
epoch 5:  train 0.1890, test 0.1865
epoch 10: train 0.1855, test 0.1850
epoch 15: train 0.1955, test 0.1915
epoch 20: train 0.1895, test 0.1800
epoch 25: train 0.1850, test 0.1895
epoch 30: train 0.1905, test 0.1840
best test accuracy: 0.1915
```

The run quickly rose above chance but then plateaued around `18-19%` test
accuracy. Prediction counts remained highly concentrated, mainly in classes
`0`, `1`, and `8` by epoch 30. Loss stayed high, though it drifted down slightly
from the initial evaluation.

The useful result here is mostly operational: 20 minima samples with 50 post-kick
sweeps is not the bottleneck at this subset size. With 32 workers and preallocated
manager state, 2000 training examples run in about 6-7 seconds per epoch after
warmup.

## 2026-06-09 Full Balanced Loose-Relaxation Run

After the subset run above, I reran the same configuration on the full balanced
MNIST splits:

```text
experiments/current/full_loose_progress_5421pc_20ep
```

Settings:

```text
workers = 32
train/test/train-eval per class = 5421 / 892 / 5421
train samples = 54210
test samples = 8920
epochs = 20
batchsize = 128
eval_every = 1
temp = 1e-4
lr = 0.001
sweeps = 25
covariance_samples = 20
covariance_sample_sweeps = 50
covariance_kick_steps = 25
covariance_noise_temp_factor = 4
reward = logprob
```

Timing:

```text
epoch 1: 185.65s
epoch 2: 176.72s
epoch 3: 186.93s
epochs 13-20: about 163-167s each
```

Full train/test evaluation was run after every epoch. Evaluation itself was
small compared with training: about `4-13s` for the full train split and less
than `1s` for the full test split after startup.

Accuracy:

```text
epoch 0:  train 0.09122, test 0.09462
epoch 1:  train 0.18742, test 0.18677
epoch 6:  train 0.18432, test 0.18868
epoch 10: train 0.18635, test 0.18868
epoch 15: train 0.18458, test 0.18621
epoch 20: train 0.18685, test 0.18576
best full-test accuracy: 0.18868
```

The full-data run confirms that the `25 + 50` loose-relaxation setting does not
continue learning with more epochs. It jumps from chance to roughly `18-19%` and
then stays there.

The dominant failure mode is prediction collapse. After epoch 1, the test
predictions are almost entirely in classes `1`, `4`, `6`, and `7`. By epoch 20
the prediction counts are:

```text
0-1598-0-0-4190-0-1580-1552-0-0
```

Parameter norms grow steadily through the run:

```text
w_norm:       0.49 -> 19.83
b_norm:       0.00 -> 16.39
w_input_norm: 1.53 -> 868.18
```

So the optimizer is applying a strong signal, but it is not a useful
class-balanced signal. This is now clearly a learning-rule/objective issue, not
just a subset-size issue.
