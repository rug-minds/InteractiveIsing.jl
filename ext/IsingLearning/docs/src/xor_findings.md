# XOR Findings

These notes summarize the current XOR direction in `IsingLearning`. They are
intended to prevent us from repeating the old single-equilibrium tuning loop.

## Current Successful Baseline

The current strongest baseline is the honest two-input discrete XOR run:

```text
ext/IsingLearning/examples/xor_statistical_ep_2input.jl
ext/IsingLearning/runs/xor_2input_discrete_T05_seed13/
```

Full run documentation is in
[`xor_2input_discrete_success.md`](xor_2input_discrete_success.md).

Short version:

- input is the real two-bit code, not four-way one-hot:
  `[-1, -1]`, `[-1, +1]`, `[+1, -1]`, `[+1, +1]`;
- output is direct bipolar one-hot: false `[+1, -1]`, true `[-1, +1]`;
- architecture is `2 -> 16 -> 2`;
- state set is discrete `{-1, +1}`;
- dynamics is `IsingMetropolis()` at `T = 0.5`;
- Hamiltonian is `Bilinear + MagField + Clamping`;
- there is no local potential and no double well;
- training uses the verified symmetric EP gradient, not the target-free
  surrogate;
- free and nudged phases use `1000` Metropolis proposals each;
- gradients are averaged over `Minit = 8` random starts per sample;
- validation averages over `64` random starts.

That run moved from MSE `1.11975`, accuracy `0.25` at epoch 0 to logged MSE
`0.000244`, accuracy `1.0` around epochs 3500 to 4000. It saved the trained
graph as:

```text
ext/IsingLearning/runs/xor_2input_discrete_T05_seed13/xor_statistical_ep_2input_trained_graph.jld2
```

What made it work:

- discrete spins match the desired discrete output extremes;
- finite temperature was essential: very low temperature froze the Markov
  chain, while `T = 0.5` kept the plus/minus response alive;
- repeated-state statistics made the gradient much less dependent on one
  stochastic trajectory;
- direct output clamping avoided the earlier readout/clamping mismatch;
- the verified EP sign was used.

Remaining caveat: this proves the architecture and learning rule can solve the
honest XOR case, but seed robustness is not solved yet.

## Continuous Langevin Baseline

Continuous Langevin can also solve the honest two-input XOR task, but it needed
a different effective-temperature regime from the discrete Metropolis run.

Full run documentation is in
[`xor_2input_langevin_success.md`](xor_2input_langevin_success.md).

Successful run:

```text
state = continuous
dynamics = BlockLangevin(adjusted=false, stepsize=0.1, block_size=8)
init_mode = zero
T = 0.001
β = 1.0
lr = 0.01
weight_decay = 0
free/nudged relaxation = 1000 / 1000
Minit = 4
eval repeats = 32
```

It reached logged MSE `0.0268`, accuracy `1.0` at epoch 1500, and restored
best MSE about `0.038`, accuracy `1.0`.

The earlier conclusion that Metropolis "worked better" was incomplete. The
failed Langevin comparison used the Metropolis temperature `T=0.5`, random
continuous initial states, smaller stepsize, smaller LR, and weight decay. In
continuous bounded states that kept output means weak and output variance high.

Current interpretation:

- for discrete spins, `T=0.5` is a useful finite-temperature regime;
- for continuous bounded Langevin, the same `T` is far too noisy for this
  learned field scale;
- the relevant quantity is closer to `T / typical |J|`;
- weight growth and colder dynamics are needed to make the output states sit
  near the target corners;
- zero initialization reduced basin variance enough for this small all-to-all
  test.

## Active Training Path

The path used by the current examples is:

```text
init_mnist_trainer
-> _worker_process
-> Forward_and_Nudged
-> contrastive_gradient
-> Optimisers.update
```

The old `ComputeGradients.jl`, `ChainRules.jl`, and `ep_train_step!` path is not
part of these experiments.

## Gradient Sign

This is the sign convention to keep unless the core Hamiltonian convention
changes.

1. The samplers minimize a Hamiltonian `H`.
2. Positive clamping strength means the nudged dynamics minimize
   `H(θ, s) + β C(s, y)`.
3. Negative clamping strength means the anti-nudged dynamics minimize
   `H(θ, s) - β C(s, y)`.
4. Equilibrium propagation is often written with a primitive `Φ` that the
   dynamics maximizes. In this codebase, `Φ = -H`.
5. The symmetric estimator for the loss gradient to pass to
   `Optimisers.update` is therefore:

   ```math
   \nabla_\theta L
   \approx
   \frac{\partial_\theta H(\theta, s_\beta)
         - \partial_\theta H(\theta, s_{-\beta})}{2\beta}.
   ```

6. `Optimisers.update` applies descent, so this is the object that should be
   supplied as the gradient.
7. `ext/IsingLearning/src/Gradient.jl` implements this by accumulating
   parameter derivatives from the plus state and subtracting derivatives from
   the minus state before the batch collector divides by `2β`.

Scalar analytic check against the code convention: for `H = s^2 - b s`,
`C = 1/2 (s - y)^2`, `b = 0.2`, `y = 1`, and `β = 0.01`, the code gives
`dL/db ≈ -0.450011`; the exact derivative is `-0.45`.

## Why The Old XOR Path Was A Dead End

The old XOR debugging stack optimized one or a few relaxed states from a hard,
bounded, mostly linear Ising system. That is a fragile target:

- Bounded linear units tend to behave like threshold units near low temperature.
- Small output nudges often do not produce a smooth hidden response.
- When hidden units do respond, the response can be a discontinuous switch
  rather than a useful infinitesimal susceptibility.
- Accuracy could briefly reach `1.0` while margins stayed tiny and MSE stayed
  high, so accuracy alone was not a reliable success criterion.
- Longer runs and larger relaxation budgets improved some transients but did
  not make the single-state objective robust.

The conclusion is not that EP has the wrong sign. The conclusion is that the
single-equilibrium hard-Ising XOR setup is probably the wrong first debugging
target.

## Current Direction

The first statistical XOR example was `examples/xor_statistical_ep.jl`.

It intentionally simplifies the task encoding:

- Input is four bipolar one-hot units, one for each XOR input case.
- Output is two bipolar one-hot units: false is `[+1, -1]`, true is `[-1, +1]`.
- The graph is all-to-all input-hidden-output.
- There is no local potential and no double well in the default setup.
- Direct `Clamping` is valid because the task loss is directly on the two
  output spin values.

The important change is statistical averaging:

- Each training sample is run from multiple initial states.
- The contrastive gradients are averaged over those repeated relaxations.
- Evaluation uses repeated relaxations and reports mean output, output
  variance, MSE, and accuracy.

This is closer to a stochastic/statistical EP diagnostic than to the old
single-fixed-point search.

That setup remains useful as a plumbing test, but it is not an honest XOR
benchmark because the four-way one-hot input already separates the four input
cases linearly. The honest all-to-all baseline is now the two-input discrete
run documented above.

## Local Checkerboard Double-Well Probe

The local checkerboard experiment now has an optional static double well and
temperature-annealed sampler wrapper. The local potential is not trainable.

First 2x2 Langevin probes with `Minit = 4`, `500/500` free/nudged relaxation,
and nudged temperature starting near the double-well barrier did not solve the
task. The best of the two short probes ended near MSE `1.02`, accuracy `0.5`.

Documentation:

```text
ext/IsingLearning/docs/src/local_checkerboard_doublewell_anneal.md
```

Current interpretation: the mechanism runs, but the bottleneck is still weak
class-separated readout response, not just lack of local extremization.

## Current Baseline

The first useful baseline uses a cold but still finite temperature:

```text
ISING_XOR_STAT_TEMP=0.001
ISING_XOR_STAT_FREE_RELAXATION=50
ISING_XOR_STAT_NUDGED_RELAXATION=50
ISING_XOR_STAT_MINIT=4
ISING_XOR_STAT_EVAL_REPEATS=16
ISING_XOR_STAT_EPOCHS=500
```

With the default architecture `4 -> 16 -> 2`, a 500-epoch probe moved from MSE
about `1.00` to about `0.24`, with accuracy reaching `1.0` around epoch 150.

A 1000-epoch probe with the same settings restored a best point with MSE about
`0.089` and accuracy `1.0`. That crosses the first practical target for the
simplified XOR task. Output variance is still noticeable, especially on one
true case, so the next target is robustness over more validation repeats and
seeds.

Artifacts from the first successful probes:

- `ext/IsingLearning/runs/xor_statistical_ep_probe_500_cold/`
- `ext/IsingLearning/runs/xor_statistical_ep_probe_1000_cold/`

Current Langevin sanity recheck:

- `ext/IsingLearning/runs/xor_langevin_onehot_relax1000_recheck_20260510_031327/`
- same simple one-hot all-to-all `4 -> 16 -> 2` task;
- `BlockLangevin(adjusted=false, stepsize=0.05, block_size=8)`;
- `T=0.001`, `Minit=4`, `eval_repeats=16`;
- free/nudged relaxation increased to `1000/1000`;
- restored MSE `0.016789`, accuracy `1.0`.

The old `50/50` relaxation setting no longer reproduces the one-hot success in
the current code. Treat `1000/1000` as the current Langevin sanity-check
setting before tuning harder encodings.

## Multiplexed Pattern Probes

The distributed 4x4 pattern code works when the input codewords are actually
mutually orthogonal.

The failed first version used `0.5 * (±vertical + ±horizontal)`. Those are
distinct multiplexed patterns, but they are not all pairwise orthogonal: cases
that differ in both signs are anti-correlated. That version lowered MSE but
collapsed the output margins to mostly the same sign.

The corrected `examples/xor_multiplexed_patterns.jl` uses four Walsh-style
4x4 input codewords:

- constant
- vertical
- horizontal
- checkerboard = vertical * horizontal

The file now checks this at runtime:

```text
input_gram_offdiag = 0.0
output_dot = 0.0
```

The output targets are direct 4x4 patterns: XOR false is the vertical pattern
and XOR true is the horizontal pattern. The loss is direct output-spin MSE, and
classification is nearest output pattern.

Corrected all-to-all result:

- architecture: `4x4 -> 4x4 -> 4x4`
- 1500 epochs, otherwise the same statistical settings as the successful
  two-output run
- MSE went from about `1.05` to logged `0.068`
- restored best evaluation gave MSE about `0.079`
- accuracy was `1.0`

Artifact:

- `ext/IsingLearning/runs/xor_multiplexed_patterns_orthogonal_1500/`

The square-local file is `examples/xor_conv_square_patterns.jl`.

The current implementation keeps the normal worker/composite training path but
uses a specialized index set for the embedded code:

- input layer: `16x16`
- hidden layer: configurable; the working run uses one `16x16` hidden layer
- output layer: `16x16`
- the 4x4 input/output code is embedded on `[3, 7, 11, 15]`
- input code sites are frozen during free/nudged phases
- only output code sites are sampled; non-code output sites are zeroed after
  input application so they do not act as fixed random fields
- adjacent layers use random square-local connections

Important implementation fixes:

- `Coordinate` iteration now iterates over `c.coords.I`. This preserves the
  intended semantics and fixes the non-periodic same-layer `WeightGenerator`
  path that surfaced in the square-local example.
- The example uses `WeightGeneratorOld` for captured RNG same-layer generators
  because `DirectMethod` cannot reliably introspect those anonymous keyword
  closures on the current Julia path.
- `apply_input` has an optional `:after_apply_input!` graph-addon hook. Existing
  graphs are unchanged; the square-local example uses it to zero inactive
  embedded output sites after every reset/input write.

Dead ends checked:

- Two hidden `16x16` layers with random same-layer NN weights did not separate
  the XOR classes in short probes.
- Keeping random same-layer NN weights in the one-hidden setup could reach
  accuracy `1.0`, but parameter growth was unstable and MSE stayed high.
- One-hot input codes did not improve the local square run; the orthogonal
  Walsh-style code is not the main blocker.
- Simply increasing relaxation helps MSE but can hurt class separation if the
  recurrent random substrate is still present.

Best square-local result so far:

```text
ISING_XOR_CONV_HIDDEN_LAYERS=1
ISING_XOR_CONV_INTERNAL_WEIGHT_SCALE=0.0
ISING_XOR_CONV_BIAS_SCALE=0.0
ISING_XOR_CONV_FREE_RELAXATION=200
ISING_XOR_CONV_NUDGED_RELAXATION=200
ISING_XOR_CONV_BETA=0.2
ISING_XOR_CONV_LR=0.001
ISING_XOR_CONV_WEIGHT_DECAY=0.001
ISING_XOR_CONV_GRAD_CLIP=30
ISING_XOR_CONV_MINIT=4
ISING_XOR_CONV_EVAL_REPEATS=32
```

This run keeps local inter-layer connectivity and the embedded 4x4 multiplexed
input/output code, but disables random same-layer recurrent weights. It reached
accuracy `1.0` and logged MSE about `0.104` at 12k epochs. The restored
checkpoint reevaluation was MSE about `0.122`, accuracy `1.0`, which indicates
remaining validation variance rather than a sign error.

Artifacts:

- `ext/IsingLearning/runs/xor_conv_nointernal_relax200_beta02_12000/`

Interpretation: the square-local architecture can learn XOR, but the fully
recurrent random in-layer substrate is too noisy for the present EP estimator.
For now, use local inter-layer paths first, then reintroduce internal
connections gradually with normalization or a much smaller scale.

## Hidden Internal Connection Probe

The successful square-local file and trained graph were snapshotted here:

```text
ext/IsingLearning/runs/snapshots/xor_conv_square_success_20260508_105831/
```

To test hidden same-layer connectivity without also perturbing the input and
output substrates, `examples/xor_conv_square_patterns.jl` now has separate
internal scales:

```text
ISING_XOR_CONV_INPUT_INTERNAL_WEIGHT_SCALE
ISING_XOR_CONV_HIDDEN_INTERNAL_WEIGHT_SCALE
ISING_XOR_CONV_OUTPUT_INTERNAL_WEIGHT_SCALE
```

The tested setup kept input/output internal scales at `0.0` and enabled only
hidden-layer NN connections.

Results after 5000 epochs with the same stable square-local settings:

```text
hidden internal scale 0.005  -> restored MSE about 0.360, accuracy 1.0
hidden internal scale 0.001  -> restored MSE about 0.331, accuracy 1.0
hidden internal scale 0.0001 -> restored MSE about 0.386, accuracy 1.0
no hidden internal edges      -> restored MSE about 0.138 at 5000, accuracy 1.0
```

The embedded-pattern architecture should also give the input and output
substrates internal connectivity, not only the hidden layer. Two all-substrate
runs were checked with `NN = 5`, one hidden layer, input/hidden/output internal
connections enabled, and the same 200/200 relaxation settings:

```text
all internal scales 0.001  -> restored MSE about 0.308, accuracy 1.0
all internal scales 0.0001 -> restored MSE about 0.410, accuracy 1.0
```

So same-layer internal connections across input/hidden/output do not make the
task impossible, but with the current random initialization and shared optimizer
they still hurt output-vector MSE relative to the no-internal-edge control. This
is not evidence against interconnected layers in general; it means random
same-layer recurrent weights need additional control before they help. The next
reasonable step is to add a separate normalization/decay policy for internal
weights, or initialize internal edges near zero and train them with a smaller
learning-rate multiplier than inter-layer weights.

## Success Criteria

Use MSE and robustness, not only classification:

- First useful signal: mean-output MSE decreases from initialization.
- Practical target: accuracy `1.0` and output MSE below `0.1` over repeated
  initial states.
- Also watch output standard deviation and free-to-nudged response norm. A
  low-MSE result with high output variance is not yet robust.

## Local Checkerboard Output Pattern Clamping

The local checkerboard experiment now has an experiment-local
`OutputPatternClamping` term. It nudges only the physical output-code indices:

```math
H_\mathrm{out} = \frac{\beta}{2}\sum_k (s_{i_k} - y_k)^2
```

This avoids the generic full-graph `Clamping` problem where non-output states
would silently be pulled toward zero when the target vector only fills the
output layer.

The first 2x2 double-well Langevin run with direct output-pattern clamping did
not improve learning: MSE stayed around `1.1` and accuracy did not exceed random
levels by the end. That means the scalar readout clamp was not the only failure
mode in the local checkerboard setup. Keep the output-pattern term, but the
next local experiments should focus on coupling scale/normalization,
temperature scale, and possibly a different minimization sampler.

A no-double-well control with the same output-pattern clamping also failed
(`pattern_clamp_nodw_langevin_20260510_201245`, final MSE about `1.26`,
accuracy `0.5`). So for now the double well is not the explanation and should
be ignored while tuning the local checkerboard setup.

A broader no-double-well grid was run next. The best short screen was
BlockLangevin with `T=0.01`, `stepsize=0.08`, `beta=0.2`, and inter-layer scale
`0.25`, reaching accuracy `1.0` and MSE about `0.65`. Longer reruns did not
drive the MSE down; seed-matched longer training stayed around MSE `0.8-0.9`.
Stronger coupling up to `J=1.0`, no weight decay, and larger LR also failed to
push the output amplitudes toward `+/-1`.

Current conclusion for local checkerboard: signs can sometimes separate, but
the physical output pattern remains weak. The next change should probably be
architectural or estimator-related, not another tiny tweak to the double-well
or scalar-vs-pattern clamping.

The latest locality scan checked both same-layer `internal_nn` and inter-layer
`inter_radius`. Short screens found temporary sign-correct points, e.g.
`internal_nn = 2, inter_radius = 1.01` at `T = 0.01`, `stepsize = 0.08`,
`beta = 0.2`, and `J = 0.25`, with best MSE about `0.67`. A more averaged
rerun did not hold up: `Minit = 4` and `150/150` relaxations stayed near MSE
`0.95` and did not reach accuracy `1.0`. So both locality knobs matter, but the
current local setup still lacks a robust amplification mechanism.

Next local-checkerboard work should normalize coupling scales by fanout/degree,
check that learned adjacency remains symmetric after optimizer updates, and
try discrete Metropolis as a control for the continuous Langevin relaxation.
