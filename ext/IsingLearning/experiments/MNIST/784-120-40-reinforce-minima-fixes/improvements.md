# REINFORCE Minima Fixes

This folder keeps the REINFORCE-style minima-learning trials separate from the
earlier `784-120-40-surrogate-attractor` folder.

Script:

```text
mnist_784_120_40_reinforce_minima_fixes_adam.jl
```

## Implemented Fixes

### 1. Per-label reward baseline

The previous rule subtracted one minibatch-wide reward baseline:

```text
grad -= mean_reward_batch * mean_features_batch
```

This can let easy or overrepresented predicted classes dominate the covariance
signal even when the input labels are balanced.

The new `ISING_MNIST_IF_REWARD_BASELINE=label` mode accumulates feature sums and
reward sums separately for each target digit, then subtracts the baseline for
the digit that generated the sample:

```text
grad_label -= mean_reward_for_target_label * feature_sum_for_target_label
```

The baseline depends on the label/input, not on the sampled minimum, so it is a
valid REINFORCE-style variance-reduction baseline.

### 2. Input projection normalization

The old full run had:

```text
w_input_norm: 1.53 -> 868.18
```

That made the external input field dominate the recurrent Ising dynamics. The
new `ISING_MNIST_IF_W_INPUT_NORMALIZATION=row` mode normalizes each hidden-unit
input vector after every Adam step.

For the run below:

```text
ISING_MNIST_IF_W_INPUT_ROW_NORM=0.14
```

The total `w_input_norm` stayed near:

```text
1.5336
```

throughout the run.

### 3. Output bias class-prior projection

The new `ISING_MNIST_IF_PROJECT_OUTPUT_BIAS_PRIOR=true` mode removes the mean
bias-gradient component within each digit's output replicas. This prevents the
output bias update from simply learning a global class prior for balanced MNIST.

### 4. Margin reward

The run used:

```text
ISING_MNIST_IF_ATTRACTOR_REWARD=margin
```

with reward:

```text
target_score - max(non_target_scores)
```

This directly rewards separating the target class from the strongest wrong
class, instead of rewarding only the current log-probability-like score.

### 5. Explicit sparse symmetry constraint

The Ising coupling matrix must remain symmetric. The new script explicitly
symmetrizes the stored sparse `w` vector:

```text
before Adam:  symmetrize gradient.w
after Adam:   symmetrize params.w
on restore:   symmetrize checkpoint params.w before syncing
```

It also records:

```text
symmetry_error
grad_symmetry_error
```

in the CSV. In the run below both stayed exactly `0.0`.

### 6. Recurrent coupling norm projection

The recurrent Ising coupling scale is mostly interchangeable with temperature:

```text
E / T
```

so letting all recurrent weights grow mainly changes the effective temperature
scale. The script now supports:

```text
ISING_MNIST_IF_W_NORMALIZATION=global
ISING_MNIST_IF_W_NORM=<target norm>
```

When enabled, the sparse recurrent `w` vector is projected back to the requested
global norm after symmetrization and after each Adam update. This keeps the
coupling matrix symmetric and prevents pure scale growth from dominating the
learning dynamics.

Smoke test:

```text
ISING_MNIST_IF_W_NORMALIZATION=global
ISING_MNIST_IF_W_NORM=1.0
```

gave:

```text
epoch 0 w_norm = 1.0
epoch 1 w_norm = 0.99999994
symmetry_error = 0.0
grad_symmetry_error = 0.0
```

## Full Balanced Run

Run directory:

```text
experiments/current/full_labelbaseline_margin_rowwinput_symmetric_5421pc_20ep
```

Settings:

```text
workers = 32
train/test/train-eval per class = 5421 / 892 / 5421
batchsize = 128
scheduled epochs = 20
stopped after completed epoch = 19
temp = 1e-4
lr = 0.001
sweeps = 25
covariance_samples = 20
covariance_sample_sweeps = 50
covariance_kick_steps = 25
covariance_noise_temp_factor = 4
reward = margin
reward_baseline = label
project_output_bias_prior = true
w_input_normalization = row
w_input_row_norm = 0.14
```

The run was stopped during epoch 20 because accuracy had clearly plateaued and
then degraded.

## Results

Previous full balanced loose-relaxation run, without these fixes:

```text
best full-test accuracy = 0.18868
final epoch 20 test accuracy = 0.18576
w_input_norm = 1.53 -> 868.18
prediction collapse mostly into classes 1, 4, 6, 7
```

New fixed run:

```text
epoch 0:  train 0.09006, test 0.09170
epoch 1:  train 0.21040, test 0.20930
epoch 2:  train 0.20517, test 0.20874
epoch 4:  train 0.20705, test 0.21244
epoch 8:  train 0.20177, test 0.20460
epoch 14: train 0.18956, test 0.19159
epoch 17: train 0.18187, test 0.18901
epoch 19: train 0.14483, test 0.14507
best full-test accuracy = 0.21244
```

The fixes improved the best full-test accuracy from `0.18868` to `0.21244`.
They also prevented the `w_input` norm explosion and kept the sparse matrix
exactly symmetric.

However, the run still did not sustain learning. After the early improvement,
accuracy drifted downward. By epoch 19 the prediction distribution had degraded:

```text
1157-1130-1183-11-1142-11-1151-758-1175-1202
```

This is not the old collapse into only a few classes, but it is still a bad
classifier: classes 3 and 5 nearly disappear, while most other classes are
overused.

## Takeaways

The four fixes helped, but they did not solve the plateau.

What they did fix:

- `w_input` no longer explodes.
- The sparse recurrent coupling storage remains exactly symmetric.
- The early best accuracy is higher than the previous full-data run.
- Early prediction counts are more class-diverse than the old collapse.

What remains broken:

- Accuracy is still low and unstable.
- The rule still degrades after several epochs.
- Prediction balance is not preserved.
- The recurrent norms continue growing even with normalized input projection.

## Next Things To Try

The most plausible next changes are:

```text
1. run with recurrent w projection, probably w_norm in {0.5, 1.0, 2.0}
2. lower lr, probably 0.0003 or 0.0005
3. keep label baseline + row w_input normalization + symmetry
4. compare margin reward against logprob reward under the same constraints
5. add an explicit prediction-balance diagnostic/loss term only if needed
```

The current result suggests that normalization and label baselines are useful,
but the recurrent update still needs regularization or a lower learning rate.
