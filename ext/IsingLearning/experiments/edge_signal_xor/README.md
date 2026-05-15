# Edge Signal XOR Experiment

## Current File Layout

The active split-edge path is now:

```text
edge_signal_split_input_core.jl
run_split_input_high_sweeps.jl
```

`edge_signal_split_input_core.jl` is the single source file for the active
split-input architecture and training helpers. It is source-only and does not
start a run by itself.

`run_split_input_high_sweeps.jl` is the current configuration/runner file. It
includes only `edge_signal_split_input_core.jl`.

Older exploratory scripts were moved to:

```text
legacy/
```

Those files are reference material only. They still reflect the earlier
exploration history and should not be used as the active entrypoint.

The active split-input graph is:

```text
2 input spins -> split first hidden edge -> 8x8 hidden layer -> last hidden edge -> 1 output spin
```

The current active runner uses the existing threaded trainer path
(`init_mnist_trainer` / `_run_minibatch!`), not `ProcessManager`. So there is no
manager ownership/synchronization claim in the current split-edge run. If this
experiment is moved to `ProcessManager`, it should be built from
`edge_signal_split_input_core.jl` directly, with persistent manager-owned
workers and no separate external worker vector.

## Legacy Baseline

This folder originally contained one experiment file:

- `edge_signal_xor.jl`

The graph is:

```text
2 input spins -> left edge of 8x8 hidden layer -> right edge -> 1 output spin
```

The hidden layer has local symmetric couplings controlled by `EDGE_XOR_HIDDEN_NN`.
The input and output connections are edge-only, not all-to-all.

Run a quick learning search without response traces:

```julia
ENV["EDGE_XOR_SKIP_RESPONSE"] = "true"
ENV["EDGE_XOR_EPOCHS"] = "500"
ENV["EDGE_XOR_NNS"] = "0,1,2"
include("ext/IsingLearning/experiments/edge_signal_xor/edge_signal_xor.jl")
```

Run learning plus response traces:

```julia
ENV["EDGE_XOR_SKIP_RESPONSE"] = "false"
ENV["EDGE_XOR_RESPONSE_PRE_SWEEPS"] = "60"
ENV["EDGE_XOR_RESPONSE_SWEEPS"] = "80"
include("ext/IsingLearning/experiments/edge_signal_xor/edge_signal_xor.jl")
```

The response trace starts from a relaxed source XOR input, switches to a target
XOR input, and logs after every full sweep. It records output response, left and
right edge response, total hidden response, and the average response column.

The smoke test verified:

- The adjacency is symmetric.
- Each input spin connects to 8 hidden spins on the left edge.
- The output spin connects to 8 hidden spins on the right edge.
- Hidden local couplings are generated through `@WG`.

First checked run:

- folder: `runs/edge_signal_xor_20260512_151913`
- `NN=1`, 500 epochs, `Minit=4`, free/nudged `400/400`
- best logged MSE was `0.856821` at epoch `200`, accuracy `0.75`

That run did not solve XOR. It is useful as a baseline because it confirms the
edge-only graph trains through the existing IsingLearning path and produces the
response trace files, but the recipe still needs tuning before comparing learned
versus random signal propagation.

Second checked run:

- folder: `runs/edge_signal_xor_20260512_153558`
- `NN=5`, 1000 epochs, `Minit=4`, free/nudged `400/400`
- hidden layer was made non-periodic for this run, which is the correct boundary
  condition for an edge-propagation experiment
- best logged MSE was `0.860759` at epoch `200`, accuracy `0.75`
- final MSE was `1.022974`, final accuracy `0.5`

This exact `NN=5` recipe did not work. It initially had some class separation,
then lost it. That suggests wider local hidden coupling alone is not enough; the
next useful probes are temperature fraction, input/output edge coupling scale,
and relaxation length.

Targeted grid from `runs/edge_signal_grid_20260512_154608`:

| hidden NN | T fraction | edge scale | hidden scale | best MSE | best accuracy | best epoch |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.015 | 0.12 | 0.04 | 0.650530 | 1.0 | 400 |
| 1 | 0.025 | 0.16 | 0.04 | 0.604403 | 1.0 | 300 |
| 1 | 0.050 | 0.16 | 0.04 | 0.614388 | 1.0 | 200 |
| 2 | 0.015 | 0.16 | 0.025 | 0.680686 | 1.0 | 300 |
| 2 | 0.025 | 0.16 | 0.025 | 0.529323 | 1.0 | 400 |
| 2 | 0.050 | 0.20 | 0.025 | 0.561450 | 1.0 | 200 |
| 3 | 0.025 | 0.20 | 0.015 | 0.586935 | 1.0 | 300 |
| 3 | 0.050 | 0.20 | 0.015 | 0.609628 | 1.0 | 400 |
| 5 | 0.025 | 0.20 | 0.008 | 0.760501 | 1.0 | 200 |
| 5 | 0.050 | 0.20 | 0.008 | 0.801339 | 1.0 | 100 |
| 5 | 0.025 | 0.25 | 0.005 | 0.649038 | 1.0 | 1 |
| 5 | 0.080 | 0.25 | 0.005 | 0.711917 | 1.0 | 1 |

The best tested point is `NN=2`, temperature fraction `0.025`, edge scale
`0.16`, hidden-local scale `0.025`. It reached `1.0` accuracy but not low MSE.
The output signs are learned, but the output magnitudes remain small. The next
step should test whether this is caused by weak output fields, finite
temperature, or insufficient relaxation by logging raw output samples and
effective output fields.

Longer validation check:

- folder: `runs/edge_signal_xor_20260512_160432`
- same best grid region: `NN=2`, T fraction `0.025`, edge scale `0.16`,
  hidden-local scale `0.025`
- training free/nudged stayed `500/500`
- validation relaxation was increased to `3000`
- result did not improve: final MSE `1.036243`, final accuracy `0.25`

This means the previous MSE around `0.53` was not simply because validation was
too short. Longer relaxation from random validation starts can settle into
different attractors or erase the weak sign separation. The next diagnostic
should compare validation from fixed warm starts versus random starts, and log
the raw output samples rather than only their mean.

Single-relaxation snapshot diagnostic:

- folder: `runs/relaxation_snapshots_20260512_162128`
- graph: best graph from `runs/edge_signal_grid_20260512_154608/05_NN2_T0p025_io0p16_h0p025_lr0p002/best_graph.jld2`
- input case: first XOR case
- dynamics: unadjusted `LocalLangevin`, stepsize `0.4`
- saved snapshots every `10` full sweeps up to `500` full sweeps

The state was not frozen after 500 full sweeps. Consecutive saved snapshots in
the late part of the run, from 400 to 500 full sweeps, had:

| interval | value |
|---|---:|
| average cosine similarity | `0.956254` |
| minimum cosine similarity | `0.942140` |
| maximum cosine similarity | `0.970813` |
| average relative change norm | `0.294919` |
| relative change norm range | `0.246095` to `0.340104` |

The output spin also kept moving. Over the whole 500-sweep trajectory it ranged
from `-0.982753` to `-0.333840`, and the final output was `-0.839982`.

So the issue is not just a too-short 500-sweep validation run. With this
finite-temperature Langevin setup the trajectory keeps fluctuating around the
low-energy region instead of becoming nearly static. If we want a deterministic
energy-minimization diagnostic, we need a separate low-noise or zero-noise
validation mode rather than interpreting finite-temperature Langevin samples as
frozen states.

## Stable LocalLangevin Retest

The edge-signal experiment was rerun with the updated `LocalLangevin` sampler:

```text
runs/stable_local_langevin_edge_20260514
```

Settings:

| parameter | value |
|---|---:|
| hidden NN values | 1, 3, 5 |
| temperature fraction | 0.05 |
| stepsize | 0.8 |
| beta | 1.0 |
| learning rate | 0.0008 |
| free relaxation | 300 |
| nudged relaxation | 300 |
| validation relaxation | 600 |
| random starts per sample | 6 |
| validation repeats | 16 |
| input-hidden scale | 0.16 |
| hidden-output scale | 0.16 |
| hidden-local scale | 0.02 |

Results:

| hidden NN | best MSE | best accuracy | best epoch |
|---:|---:|---:|---:|
| 1 | 0.812881 | 0.75 | 240 |
| 3 | 0.768792 | 0.75 | 240 |
| 5 | 0.792204 | 0.75 | 0 |

This did not solve the edge-propagation task. The best point was `NN=3`, but it
still only reached `0.75` accuracy and MSE around `0.77`. The failure mode is
the same as earlier local runs: signs can partially separate, but output
magnitudes stay weak and one XOR case remains unreliable.

## 2026-05-14 Edge Retests

After the scalar `2 -> 4 -> 1` LocalLangevin XOR run started working with full
`±1` targets and smaller direct clamping, the edge experiment was retested with
the same lesson.

### Low-Beta 8x8 Edge Search

Run folder:

```text
runs/edge_lowbeta_20260514_131047
```

This kept the original scalar-output edge architecture:

```text
2 input spins -> left edge of 8x8 hidden layer -> right edge -> 1 output spin
```

The tested settings used `β = 0.20` or `β = 0.35`, `LocalLangevin` stepsize
`0.8`, temperature fractions `0.035` or `0.05`, and hidden `NN = 1, 2, 3`.

Best row:

| hidden NN | beta | T fraction | edge scale | hidden scale | best MSE | best accuracy | best epoch |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.20 | 0.05 | 0.20 | 0.015 | 0.781474 | 1.0 | 720 |

This reached correct signs at some checkpoints, but the output means were very
small, approximately `[-0.203, 0.050, 0.165, -0.056]` at the best logged point.
So the failure was not just that old `β = 1` or `β = 2` was too strong. Smaller
`β` can recover sign information, but it did not produce a strong scalar output.

### Shorter Hidden Paths

Run folder:

```text
runs/edge_small_...
```

I tried the same edge-input/readout idea on 2x2 and 4x4 hidden layers. This was
not successful. The 2x2 runs either stayed at `0.5` accuracy or collapsed to the
same output sign for every XOR case. The 4x4 rows showed the same behavior. This
means the failure is not only the eight-column propagation distance.

### Nudged-Temperature Bump

Run folder:

```text
runs/edge_nudged_temp_20260514_132912
```

The plus/minus nudged phases were run hotter than the free phase, with factors
`2`, `3`, or `4`. This was meant to test whether the clamp could escape the free
attractor before settling.

This did not help. Gradient norms became large, signs became noisy, and the best
MSE stayed around `0.68` to `0.91` depending on the row. The temperature bump was
therefore too noisy in this direct form. If revisited, it should be an actual
short bump followed by cooling inside the nudged phase, not a fully hot nudged
phase.

### Positive Hidden-Local Couplings

Run folder:

```text
runs/edge_ferro_20260514_133448
```

The hidden local couplings were initialized positive instead of signed random,
while edge input/output couplings remained signed random and trainable.

This also did not solve the task. With `NN = 1`, the output collapsed to the
same sign for all XOR cases. With `NN = 2` and `NN = 3`, output means stayed
small and stochastic. So a fully ferromagnetic hidden medium is too simple: it
propagates, but it does not maintain enough distinguishable structure for XOR.

### Output-Edge Readout

Run folder:

```text
runs/edge_output_edge_20260514_133949
```

This replaced the single scalar output spin with eight output spins and
classified by their mean sign. The input was still two raw input spins connected
to the hidden left edge; the hidden layer was still 8x8 with local connections.

This did not work either. The output-edge mean stayed near zero or took the
wrong signs. So the problem is upstream of scalar-output variance: the hidden
edge state is not becoming a reliable XOR-dependent signal under these recipes.

### Current Interpretation

The scalar `2 -> 4 -> 1` success does not transfer directly to the edge-local
graph. The current edge graph has two simultaneous difficulties:

1. The hidden layer must propagate information from left to right.
2. The same hidden layer must build a nonlinear XOR representation.

The current random local hidden couplings, positive hidden couplings,
temperature bumps, and output-edge averaging did not solve both requirements at
once. The next useful experiment should separate these requirements more
explicitly, for example by measuring whether each individual input bit can be
decoded from the right edge before asking for XOR, or by adding a second local
hidden layer so the first layer mainly propagates input and the second layer
combines features.

### Two-Class Output Edge

Run folder:

```text
runs/edge_twoclass_20260514_134907
```

This kept the raw two-spin input and the 8x8 hidden edge path, but replaced the
single scalar output with an `8x2` output layer. Classification used:

```text
mean(true output column) - mean(false output column)
```

Best row:

| hidden NN | beta | T fraction | input scale | hidden scale | output scale | best MSE | best accuracy | best epoch |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.20 | 0.025 | 0.35 | 0.005 | 0.40 | 3.472601 | 1.0 | 4500 |

The signs can be correct, but the class-score difference is still very small.
At the best logged point the scores were about:

```text
[-0.340, 0.062, 0.153, -0.009]
```

The two-class target score is `±2`, so the MSE remains large. This means adding
output capacity alone did not make the hidden right edge a strong XOR-dependent
signal.

### Split Input Halves, Scalar Output

Run folders:

```text
runs/edge_split_input_20260514_135733
runs/edge_split_input_scout_20260514_141216
runs/edge_split_input_cold_scout_20260514_141635
```

This is the current requested architecture:

```text
2 input spins -> split first hidden edge -> 8x8 hidden layer -> last hidden edge -> 1 output spin
```

The two input spins remain the only inputs. Input spin 1 connects only to the
upper half of the first hidden edge. Input spin 2 connects only to the lower
half. The output remains one scalar spin connected to the last hidden edge.

Best rows so far:

| run | hidden `NN` | beta | T fraction | hidden scale | output scale | best MSE | best accuracy | best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| first split search | 5 | 0.20 | 0.025 | 0.005 | 0.40 | 0.688475 | 0.75 | 1800 |
| scout, lower `T` | 5 | 0.20 | 0.018 | 0.005 | 0.40 | 0.575124 | 1.0 | 1800 |
| colder scout | 5 | 0.20 | 0.015 | 0.005 | 0.40 | 0.922644 | 0.5 | 1200 |

The useful result is specific: `T fraction = 0.018` got all four signs correct,
but output means were still small, roughly `[-0.20, 0.20, 0.40, -0.20]`. That
explains why accuracy reached `1.0` while MSE stayed around `0.58`. Colder runs
at `0.015` and `0.012` did not improve this; they more often froze into the
wrong sign structure. Increasing the output-edge initialization scale also did
not solve the weak-output problem in the tested runs.

Regularization and small nudged-temperature bump:

```text
runs/edge_split_input_regularized_20260514_143902
```

| trial | weight decay | nudged temp factor | best MSE | best accuracy | best epoch |
|---|---:|---:|---:|---:|---:|
| baseline `T=0.018` | 0 | 1.00 | 1.146314 | 0.25 | 3000 |
| weight decay | 0.01 | 1.00 | 0.901205 | 0.75 | 1 |
| stronger weight decay | 0.05 | 1.00 | 0.847711 | 0.75 | 0 |
| nudged temp bump | 0 | 1.25 | 0.814453 | 1.0 | 2400 |
| weight decay + temp bump | 0.01 | 1.25 | 0.828967 | 1.0 | 2400 |

Adam already supplies momentum-like first and second moment estimates. Weight
decay did not help this edge setup: it pushed the scalar output toward weak
same-sign states. A small nudged-only temperature bump did produce a transient
`accuracy = 1.0`, but the output magnitudes were still weak, so the MSE stayed
above `0.8`.

## Brainstorm For The Next Edge Attempt

The current edge files use:

```text
two input spins -> hidden left edge -> hidden right edge -> output
```

The active constraint is to keep this shape: two input spins and one scalar
output spin. The input-to-hidden map may be structured, but it should not turn
XOR into a larger one-hot input task.

Concrete next tests:

1. **Right-edge bit decoding before XOR**:
   - train or evaluate two auxiliary readouts from the right edge: one for `x1`,
     one for `x2`
   - if `x1` and `x2` cannot be decoded at the right edge, XOR cannot work there

2. **Two local hidden layers**:
   - layer 1 propagates the edge input
   - layer 2 combines propagated features and sends to output
   - this separates transport from nonlinear feature construction

3. **Lower-temperature validation only**:
   - train at finite temperature
   - evaluate at a lower temperature to check whether a weak learned sign becomes
     a strong attractor when noise is reduced

4. **Annealed nudged phase, but not fully hot**:
   - the fully hot nudged phase was too noisy
   - a better version is a short temperature bump followed by cooling inside the
     nudged phase

5. **Separate validation temperature from training temperature**:
   - the best current split-input run has correct signs but weak magnitude
   - evaluating the learned graph colder may show whether this is a readout
     noise problem or a true weak-attractor problem

## Current Clean Edge Attempts

Active source layout after cleanup:

```text
edge_signal_split_input_core.jl
run_split_input_high_sweeps.jl
run_split_input_timeavg_search.jl
run_edge_input_layout_grid.jl
```

The old exploratory files were moved to `legacy/`, including the old chained
structured/interlaced/feature runners. The active files above each include only
`edge_signal_split_input_core.jl`; active runner-to-runner include chains are
not used.

### High-Sweep Split-Input Test

Run:

```text
runs/edge_split_input_high_sweeps_clean_20260514_180651
```

This kept the raw split-edge architecture and increased relaxation to hundreds
of full sweeps. It did not solve the task:

| setting | best observed MSE | best observed accuracy | note |
|---|---:|---:|---|
| 160/160/320 sweeps | 0.868851 | 0.75 | completed |
| 240/240/480 sweeps | 1.138245 at epoch 4000 | 0.25 at epoch 4000 | stopped early |

Conclusion: more sweeps alone is not the missing ingredient.

### Time-Averaged Validation

Run:

```text
runs/edge_split_input_timeavg_20260514_191757
```

This averaged the scalar output over time after validation burn-in. One
checkpoint reached full sign accuracy, but the output magnitudes stayed weak:

| setting | best MSE | best accuracy | representative means |
|---|---:|---:|---|
| random split input, time-averaged validation | 0.591114 | 1.0 | `[-0.393, 0.184, 0.178, -0.191]` |

Colder post-hoc validation of the saved graph did not reduce MSE. The attractor
itself is weak, not merely the readout sample.

### Structured Split Input

Run:

```text
runs/edge_split_input_structured_20260514_194552
```

This removed random sign disorder in the input edge: input spin 1 drives the
upper half of the first edge with one sign and input spin 2 drives the lower
half with one sign. Nonzero trainable `MagField` initialization was added to
give the hidden layer threshold degrees of freedom.

Best observed point:

| setting | best MSE | best accuracy | representative means |
|---|---:|---:|---|
| structured input + bias | 0.527690 | 1.0 | `[-0.198, 0.191, 0.592, -0.195]` |

This is the best current edge result. It proves the signs can be learned, but
the scalar output is still too weak for low MSE.

Post-hoc scaling of the saved graph by factors `1.5` through `10` did not fix
the weak-output problem. Scaling changed the attractors and often destroyed the
sign pattern.

### Interlaced Edge Input

Run:

```text
runs/edge_split_input_interlaced_20260514_205857
```

This tested the requested edge checkerboard input. Input spin 1 drives rows
`1,3,5,7` on the first hidden edge. Input spin 2 drives rows `2,4,6,8` on the
same edge. The hidden layer still used local `NN=5` connections, matching the
best structured split-input run.

Best completed point:

| setting | best MSE | best accuracy | representative means |
|---|---:|---:|---|
| interlaced edge input | 0.603883 | 1.0 | `[-0.207, 0.002, 0.596, -0.209]` |

The interlaced input learns the correct signs early, but the scalar output
magnitude remains weak and later training drifts. It did not beat the
top/bottom structured edge input, whose best MSE was `0.527690`.

### Half-Edge Versus Interlaced Grid

Run:

```text
runs/edge_input_layout_grid_20260514_213208
```

This directly compared two input layouts with the same four parameter settings:

- `half`: input spin 1 drives rows `1:N/2`; input spin 2 drives rows `N/2+1:N`
- `interlaced`: input spin 1 drives rows `1,3,5,...`; input spin 2 drives rows
  `2,4,6,...`

All entries used `hidden NN = 5`, `Minit = 4`, `30` free sweeps, `80` nudged
sweeps, and time-averaged validation.

| layout | best MSE | best accuracy | best epoch | parameters |
|---|---:|---:|---:|---|
| half | 0.666205 | 1.0 | 1000 | `β=0.35`, `T=0.012`, `η=0.8`, `lr=0.00025`, `wd=0.003` |
| interlaced | 0.672849 | 1.0 | 1000 | `β=0.50`, `T=0.007`, `η=1.2`, `lr=0.00015`, `wd=0.005` |
| interlaced | 0.684080 | 1.0 | 3500 | `β=0.35`, `T=0.012`, `η=0.8`, `lr=0.00025`, `wd=0.003` |

In this matched grid, both layouts can reach full sign accuracy, but neither
layout gets close to the desired low scalar MSE. The half-edge layout remains
slightly better in this run, and the older structured split-input run is still
the best observed edge result at `0.527690` MSE.

### Weight Growth

Run:

```text
runs/edge_split_input_structured_growth_*
```

This removed or reduced weight decay and increased hidden-output scale. It did
not improve the best result; signs became less stable. This means simple weight
growth is not enough.

### Discrete Metropolis

Run:

```text
run_split_input_structured_metropolis.jl
```

This was stopped early because `batch_gradient` stayed exactly `0.0` through
the tested epochs in this runner. That path is not useful until the discrete
EqProp gradient path is checked separately.

### Edge Feature Bands

Run:

```text
run_edge_feature_bands.jl
```

This still used two scalar input spins, but mapped them into four row bands on
the first hidden edge:

| hidden rows | input field |
|---|---|
| 1-2 | `+x1 - x2` |
| 3-4 | `-x1 + x2` |
| 5-6 | `+x1 + x2` |
| 7-8 | `-x1 - x2` |

The nonlinear step is still the hidden Ising relaxation, not an external
one-hot input. This also reached full sign accuracy only with weak output:

| setting | best observed MSE | best observed accuracy |
|---|---:|---:|
| feature bands | 0.878582 | 1.0 |

Zero-temperature feature-band learning did not help. It tended to lock the
joint-input case to the wrong sign.

## Current Interpretation

The current scalar edge architecture can learn signs but not strong scalar
readout magnitude. The bottleneck is not relaxation length. It is the
construction of a robust local hidden feature that makes the last edge encode
"inputs differ" with enough strength for one scalar output spin.

The best next diagnostic is not another blind grid search. It is to test
whether the hidden right edge actually contains decodable information:

1. Freeze each input state.
2. Relax the graph.
3. Fit offline linear readouts from the right edge for:
   - `x1`
   - `x2`
   - `xor(x1, x2)`
4. If XOR is not linearly decodable from the right edge after relaxation, the
   one-output learner cannot succeed without changing architecture or adding a
   second local combining layer.
