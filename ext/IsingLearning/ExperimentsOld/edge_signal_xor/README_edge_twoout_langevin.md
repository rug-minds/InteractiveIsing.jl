# Edge-Connected Two-Output XOR

This note records the current edge-connected Langevin XOR results from:

```text
ext/IsingLearning/experiments/edge_signal_xor/edge_2_8x8_2_langevin.jl
```

## Setup

The file builds a graph with:

```text
2 input spins -> hidden square -> 2 output spins
```

The input spins connect only to the left edge of the hidden square:

- input spin 1 connects to the upper half of the left edge.
- input spin 2 connects to the lower half of the left edge.

Both output spins read from the whole right edge with independent signed
trainable couplings. This is important: a previous version connected each
output only to one half of the right edge, which made XOR depend almost
entirely on weak mixing inside the hidden layer.

The Hamiltonian has only:

- `Bilinear`
- trainable `MagField`
- masked direct `Clamping` on the output spins

There is no local polynomial potential.

## Best 4x4 Run Found

One `4x4` run learned cleanly:

```text
hidden = 4x4
hidden NN = 5
dynamics = BlockLangevin(adjusted=false, stepsize=0.1, block_size=8)
T = 0.001
β = 1.0
lr = 0.005
weight_decay = 0
free/nudged = 1000/1000
Minit = 4
eval_repeats = 16
input_scale = 0.4
hidden_scale = 0.2
output_scale = 0.4
bias_scale = 0.05
```

Command:

```bash
EDGE_TWOOUT_DYNAMICS=block \
EDGE_TWOOUT_HEIGHT=4 \
EDGE_TWOOUT_WIDTH=4 \
EDGE_TWOOUT_NN=5 \
EDGE_TWOOUT_EPOCHS=3000 \
EDGE_TWOOUT_LOG_EVERY=300 \
EDGE_TWOOUT_MINIT=4 \
EDGE_TWOOUT_EVAL_REPEATS=16 \
EDGE_TWOOUT_FREE=1000 \
EDGE_TWOOUT_NUDGED=1000 \
EDGE_TWOOUT_BETA=1 \
EDGE_TWOOUT_LR=0.005 \
EDGE_TWOOUT_WEIGHT_DECAY=0 \
EDGE_TWOOUT_TEMP=0.001 \
EDGE_TWOOUT_STEPSIZE=0.1 \
EDGE_TWOOUT_INPUT_SCALE=0.4 \
EDGE_TWOOUT_HIDDEN_SCALE=0.2 \
EDGE_TWOOUT_OUTPUT_SCALE=0.4 \
EDGE_TWOOUT_BIAS_SCALE=0.05 \
julia --project=ext/IsingLearning \
  ext/IsingLearning/experiments/edge_signal_xor/edge_2_8x8_2_langevin.jl
```

Result:

```text
epoch 0     MSE = 1.883697   accuracy = 0.5
epoch 300   MSE = 0.000019   accuracy = 1.0
epoch 3000  MSE = 0.000016   accuracy = 1.0
```

Run folder:

```text
ext/IsingLearning/experiments/edge_signal_xor/runs/edge_2_8x8_2_20260517_034742
```

Important: this is a real saved run, but the same nominal recipe was not
reliably reproduced in later fresh Julia processes. Later repeats often stayed
near `MSE ~= 1.0` with accuracy around `0.5-0.75`. Treat this as proof that the
edge-connected Langevin setup can solve the task, not yet as a robust recipe.

## What Helped

The best edge setup differed from the failed variants in three clear ways:

1. Zero-initialized free phases, matching the successful simple `2 -> 16 -> 2`
   Langevin recipe.
2. A two-output readout where both output spins can read the whole right edge.
   Splitting the right edge by output blocked most of the XOR information.
3. Wide enough hidden connectivity. On a `4x4` hidden layer, `NN=5` is nearly
   dense within the hidden sheet. Strict `NN=1` local connectivity did not
   learn in the tested settings.

Later stability sweeps tried lower learning rates, more averaging, lower
temperature, larger clamping, and zero temperature. These did not make the
`4x4` result reproducible. The most likely remaining issue is high stochastic
sensitivity from the unadjusted Langevin trajectories and random worker RNG
state, not the absence of a gradient signal.

## 8x8 Status

The same recipe does not yet fully solve `8x8`.

Best checked run:

```text
hidden = 8x8
hidden NN = 5
free/nudged = 2500/2500
T = 0.001
stepsize = 0.12
lr = 0.00025
input/hidden/output scales = 0.5 / 0.2 / 0.5
```

Result:

```text
best MSE = 0.267688
best accuracy = 0.9375
best epoch = 4000
```

Run folder:

```text
ext/IsingLearning/experiments/edge_signal_xor/runs/edge_2_8x8_2_20260517_035932
```

This shows a real learning signal, but not a solved `8x8` edge-local task yet.
The likely remaining issue is propagation length and optimizer stability: the
best 8x8 runs improve substantially and then drift.

## First Grid Search

Grid runner:

```text
ext/IsingLearning/experiments/edge_signal_xor/run_edge_twoout_langevin_grid.jl
```

Run folder:

```text
ext/IsingLearning/experiments/edge_signal_xor/runs/edge_twoout_grid_20260517_042735
```

Summary:

| hidden | `NN` | `T` | `η` | `lr` | `Minit` | free/nudged | scales `(input, hidden, output)` | best MSE | best accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `4x4` | `5` | `0.001` | `0.10` | `0.005` | `4` | `1000/1000` | `(0.4, 0.2, 0.4)` | `0.783175` | `0.75` |
| `8x8` | `5` | `0.001` | `0.12` | `0.00025` | `4` | `2500/2500` | `(0.5, 0.2, 0.5)` | `0.513124` | `0.875` |
| `8x8` | `5` | `0.0008` | `0.12` | `0.00015` | `8` | `4000/2000` | `(0.8, 0.1, 0.8)` | `0.996751` | `0.75` |
| `8x8` | `3` | `0.0008` | `0.12` | `0.00015` | `8` | `4000/2000` | `(0.8, 0.1, 0.8)` | `1.825946` | `0.546875` |
| `8x8` | `5` | `0.0005` | `0.10` | `0.0001` | `8` | `5000/3000` | `(1.0, 0.05, 1.0)` | `2.023628` | `0.546875` |

Interpretation:

- The useful region is still `8x8`, `NN=5`, `T=0.001`, `η=0.12`, and
  learning rate around `2.5e-4`.
- Simply making the edge couplings stronger while weakening hidden couplings
  did not help in this grid.
- Lowering temperature to `0.0005` made this setup worse, not better. The
  Langevin dynamics still seems to need thermal motion to avoid poor basins.
- `NN=3` was worse than `NN=5` in the tested settings, so the larger system
  still needs relatively broad hidden mixing.

## Runtime Clamping Beta

The edge Langevin training composite now declares `clamping_beta` as a runtime
`@input`, and the experiment-local minibatch runner calls:

```julia
run(worker; clamping_beta = config.β)
```

This replaces the earlier lexical capture of `layer.β` inside the training
routine. Validation does not use β because it runs only the free phase.

## Current Regression Check

The old `4x4` run below remains a real saved result, but the current code path
does not reproduce it. That is now verified with the same nominal parameters:

```text
hidden = 4x4
NN = 5
T = 0.001
η = 0.10
β = 1.0
lr = 0.005
free/nudged = 1000/1000
Minit = 4
```

Current result:

```text
MSE stays around 1.5-2.0
accuracy stays around 0.5
first gradient norm is about 0.11 instead of the old successful 0.51
```

The graph wiring itself was checked:

- input-to-hidden connects only the left edge.
- hidden-to-output connects only the right edge.
- hidden local couplings are symmetric.
- full adjacency is symmetric.

So the failure is not from missing edge connections.

## Symmetric vs One-Sided Gradient

The current default estimator is the symmetric EqProp-style difference:

```text
dH(s_plus)/dθ - dH(s_minus)/dθ
```

with `s_plus` produced by `+β` clamping and `s_minus` produced by `-β`
clamping.

For the current masked-output-only clamping this gives a very small effective
response on the edge XOR controls. Increasing β to `2`, `5`, or `10` did not
help; the measured gradients became smaller and the validation stayed at
random accuracy.

I added an experiment-local diagnostic mode:

```text
gradient_mode = :plus_free
```

which uses:

```text
dH(s_plus)/dθ - dH(s_free)/dθ
```

This is not the same estimator as the symmetric one, but it is useful for
diagnosis. It produces a large initial gradient (`~1.0`) and consistently
pushes MSE from about `1.8` down toward `1.0`. The opposite sign
`gradient_mode = :free_plus` is clearly wrong: it drives the MSE to about
`2.0` and keeps accuracy at chance.

Conclusion:

- The useful direction is `plus - free`.
- The symmetric `plus - minus` estimator is currently too weak for this edge
  setup.
- Even `plus - free` mostly learns outputs near zero, not robust `[-1, +1]`
  or `[+1, -1]` output states.

## Recent Temperature Grids

Recent runs:

```text
runs/edge_twoout_grid_20260518_004850
runs/edge_twoout_grid_20260518_010410
runs/edge_twoout_grid_20260518_010832
runs/edge_twoout_grid_20260518_012314
runs/edge_twoout_grid_20260518_012616
runs/edge_twoout_grid_20260518_013145
runs/edge_twoout_grid_20260518_013537
runs/edge_twoout_grid_20260518_013942
runs/edge_twoout_grid_20260518_014216
```

What was tried:

| change | result |
|---|---|
| `8x8`, two outputs, `T = 0.003` to `0.0007` | stayed near random; best MSE above `1.4` |
| `4x4`, same nominal old-success recipe | did not reproduce; stayed near random |
| high β (`2`, `5`, `10`) | gradients became smaller; no learning |
| `plus_free` one-sided estimator | MSE drops to about `1.0`, accuracy noisy and poor |
| colder validation only | sometimes improves one log point, not reproducible |
| output-output anti-coupling | worsened MSE and accuracy |
| LocalLangevin instead of BlockLangevin | no material improvement |

The strongest current statement is: this exact edge-to-edge Langevin setup has
not been made robust under the corrected masked clamping path. The old success
should not be treated as a reliable recipe until reproduced.

## Follow-Up Grids

Runtime-β grid over clamping strength:

```text
ext/IsingLearning/experiments/edge_signal_xor/runs/edge_twoout_grid_20260517_201701
```

Results were worse than the previous best region. On the `8x8`, `NN=5`,
`T=0.001`, `η=0.12`, `lr=0.00025` setup:

| β | best MSE | best accuracy |
|---:|---:|---:|
| `0.25` | `1.535379` | `0.671875` |
| `0.5` | `1.220738` | `0.640625` |
| `1.0` | `1.854748` | `0.515625` |
| `1.5` | `1.823915` | `0.515625` |

LocalLangevin comparison:

```text
ext/IsingLearning/experiments/edge_signal_xor/runs/edge_twoout_grid_20260517_202938
```

`LocalLangevin(adjusted=false)` was worse than `BlockLangevin` in the same
region. It did not learn the `4x4` or `8x8` edge setup in the tested settings.

Wider hidden connectivity:

```text
ext/IsingLearning/experiments/edge_signal_xor/runs/edge_twoout_grid_20260517_204106
ext/IsingLearning/experiments/edge_signal_xor/runs/edge_2_8x8_2_20260517_204743
```

`NN=8` was better than the comparable `NN=5` run in that grid, but it still did
not solve the task. The best `NN=8`, `hidden_scale=0.12` result reached:

```text
MSE = 0.999803
accuracy = 0.8125
epoch = 3000
```

Extending that run to `8000` epochs did not continue to a solution:

```text
best MSE = 1.061166
best accuracy = 0.671875
best epoch = 7000
```

Current interpretation: for the strict edge-to-edge `8x8` task, the present
two-spin output readout and basic Adam update are not enough. The most promising
next change is not another scalar-parameter tweak, but changing the readout to a
spatial output pattern or adding a time-averaged output statistic so the last
edge can express the class with more than two noisy spins.

## Repeated Two-Spin Output Code

The experiment file now supports:

```bash
EDGE_TWOOUT_OUTPUT_REPEATS=2
```

This changes the output target from two spins:

```text
false = [ 1, -1]
true  = [-1,  1]
```

to four spins, with two spins voting for each class:

```text
false = [ 1,  1, -1, -1]
true  = [-1, -1,  1,  1]
```

Validation now reports class accuracy by comparing the mean of the first output
group to the mean of the second output group. This matters because plain
`argmax` would treat the two repeated false spins as different labels.

Runs:

```text
ext/IsingLearning/experiments/edge_signal_xor/runs/edge_twoout_grid_20260517_215436
ext/IsingLearning/experiments/edge_signal_xor/runs/edge_2_8x8_2_20260517_215939
```

Results:

| hidden | `NN` | output code | best MSE | best accuracy |
|---:|---:|---:|---:|---:|
| `8x8` | `5` | four spins, two per class | `1.670392` | `0.5` |
| `8x8` | `8` | four spins, two per class | `1.977654` | `0.5` |
| `4x4` | `5` | four spins, two per class | `1.900524` | `0.546875` |

This repeated output code did not help. It made even the smaller `4x4` control
worse in the tested settings. The likely reason is that it adds extra output
degrees of freedom without adding a structured spatial target; the four output
spins are still just independent scalar readouts from the edge.

## Low-Beta Nudged Temperature Bump

Hypothesis checked: if `s_plus - s_minus` is too noisy, lower β should keep the
two nudged trajectories closer, while a short temperature bump at the start of
the nudged phase should help the system leave the free basin.

Implementation details in `edge_2_8x8_2_langevin.jl`:

- `nudged_temp_factor` sets the temporary nudged temperature.
- `nudged_temp_warm_fraction` controls how much of the nudged phase uses the
  temporary temperature.
- `common_nudged_rng` seeds the plus and minus branches with the same seed for
  the same sample. The seed changes by epoch, so the optimizer does not see the
  exact same noise sequence forever.

Runs:

```text
runs/edge_twoout_grid_20260518_161523
runs/edge_twoout_grid_20260518_162639
```

Results on the `4x4` control:

| β | nudged temp | lr | best MSE | best accuracy |
|---:|---:|---:|---:|---:|
| `0.05` | `3T` | `0.0002` | `1.736390` | `0.5` |
| `0.10` | `2T` | `0.0005` | `1.606674` | `0.5` |
| `0.20` | `2T` | `0.0005` | `1.650243` | `0.5` |
| `0.50` | `T` | `0.0020` | `1.485912` | `0.5` |
| `1.00` | `T` | `0.0050` | `1.497626` | `0.5` |

Interpretation: lower β plus a nudged temperature bump did not fix the
symmetric estimator. It changes the gradient scale and sometimes lowers MSE,
but it still does not produce class separation. The issue is not simply "β too
large".

## Weight Decay and Colder Validation

The `weight_decay` config was present but was not actually applied in this
experiment file. It now adds `λθ` to the local gradient before `Optimisers.Adam`
updates the parameters.

Runs:

```text
runs/edge_twoout_grid_20260518_164213
runs/edge_twoout_grid_20260518_165329
runs/edge_2_8x8_2_20260518_171659
```

Results on the `4x4` control:

| β | validation T | weight decay | best MSE | best accuracy |
|---:|---:|---:|---:|---:|
| `1.0` | `0.0005` | `0.001` | `1.095935` | `0.515625` |
| `1.0` | `0.0002` | `0.001` | `1.002882` | `0.554688` |
| `1.0` | `0.0001` | `0.001` | `1.073422` | `0.5` |
| `1.0` | `0.0002` | corrected single decay | `1.029300` | `0.515625` |

Interpretation: real weight decay helps keep MSE closer to `1.0`, but it still
does not solve classification. Colder validation does not reveal a hidden
low-noise solution; the best accuracy remains near chance.

## Current Conclusion

The corrected masked clamping path, symmetric `+β/-β` estimator, common nudged
RNG, temperature bumps, weight decay, colder validation, and coupling-scale
retuning have not reproduced the old successful `4x4` edge run.

The next checks should be structural rather than another small scalar grid:

- inspect per-case free/plus/minus output means to see whether the symmetric
  branch points in the wrong class direction or just shrinks all outputs;
- test a spatial output pattern instead of two scalar output spins;
- test an explicit time-averaged output statistic for validation and learning;
- compare against the same graph with a Metropolis sampler to isolate whether
  this is Langevin-specific or edge-readout-specific.
