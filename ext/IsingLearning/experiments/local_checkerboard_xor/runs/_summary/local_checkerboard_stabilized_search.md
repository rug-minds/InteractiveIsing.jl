# Local Checkerboard Stabilized Search

Generated: 2026-05-11

This note summarizes the isolated search implemented in:

```text
ext/IsingLearning/experiments/local_checkerboard_xor/local_checkerboard_stabilized_search.jl
```

No toolbox code was changed for this pass. The file reuses the existing local
checkerboard graph, clamping, trainer, and Processes helpers, but defines a
no-anneal worker composite for Metropolis and a few experiment-local projection
hooks.

## Brainstormed Fixes

- **No-anneal Metropolis composite.** The current annealing wrapper path does
  not compose cleanly with Metropolis in Processes. The new file builds fresh
  `IsingMetropolis()` instances for free, plus, minus, and validation phases.
- **Post-update projection.** Optional weight clipping, bias clipping/zeroing,
  and local-field normalization were added locally. These test whether training
  fails because couplings leave the temperature regime.
- **Actual old-recipe reconstruction.** The old low-MSE run used
  `temp_is_factor = true`, `inter_weight_scale = 0.25`, internal layer scales
  `0.1`, `lr = 0.003`, and specific seeds. It also used `init_mode = :zero`
  on a discrete graph. The new pass preserved the hyperparameters but used
  random init, because zero init was explicitly ruled out for new runs.
- **Short nudged phase.** Tested whether plus/minus phases lose the clamping
  perturbation if they fully relax.
- **Larger hidden layer.** Tested a 2x2 input/output code with a 4x4 hidden
  layer to see whether local capacity was the bottleneck.
- **Full bipolar input diagnostic.** Inactive input checkerboard sites were
  optionally written and frozen to `-1`. This tests whether the weak/no-input
  `(0,0)` case is the reason the local code fails.
- **Output-pattern clamping.** The scalar readout clamp was replaced by direct
  physical output-pattern clamping in several runs.
- **False-output bias prior.** One smoke probe initialized the trainable output
  biases toward the false class, to see whether `(0,0)` only needs a default
  basin.

## Results

| Probe | Best MSE | Best Acc | Interpretation |
|---|---:|---:|---|
| `metro_T0015_J012_fieldnorm`, 1200 epochs | 0.657 | 1.0 | Field normalization kept signs possible but margins too small. |
| `metro_T0015_J012_baseline`, 2500 epochs | 0.844 | 0.75 | Damped into near-zero readouts. |
| `metro_T0015_J005_long`, 6000 epochs | 0.986 | 0.5 | Small-J discrete run collapsed to exact zero readouts after the first update. |
| `metro_continuous_T0015_J005_long`, 6000 epochs | 0.979 | 1.0 | Continuous Metropolis kept nonzero gradients but margins stayed near zero. |
| `metro_T0015_J012_flipgrad`, 2500 epochs | 0.422 | 1.0 | Opposite sign gave the best number in this pass, but gradients were mostly zero and the result is not an accepted solution. |
| `metro_hot_T005_J012_strongbeta`, 2500 epochs | 0.568 | 0.75 | Hotter dynamics plus stronger beta still collapsed to zero readouts. |
| `metro_T0015_J012_baseline`, nudged 10 steps | 0.771 | 0.75 | Short nudging did not preserve a useful response. |
| `metro_oldrecipe_random_init`, 6000 epochs | 0.697 | 1.0 | The old recipe did not transfer when zero init was removed. |
| `metro_2x2_4hidden_oldrecipe_random`, 6000 epochs | 0.850 | 1.0 | Extra hidden capacity did not fix the weak-margin failure. |
| `metro_oldrecipe_persistent`, quick probe | 0.766 | 0.75 | Persistent chains immediately preferred an all-negative/readout-collapse basin. |
| `metro_oldrecipe_pattern_clamp`, 6000 epochs | 0.653 | 1.0 | Direct output-pattern clamping got signs sometimes, but margins stayed too weak. |
| `metro_oldrecipe_input_kick`, 6000 epochs | 0.578 | 0.75 | Initializing inactive input sites to `-1` helped true cases but did not fix `(0,0)`. |
| `metro_oldrecipe_full_bipolar_input`, 6000 epochs | 0.627 | 0.75 | Freezing inactive inputs to `-1` still classified `(0,0)` as true. |
| `metro_4hidden_full_bipolar_input`, 6000 epochs | 0.653 | 0.75 | Larger hidden layer did not separate the all-minus false case. |
| `metro_full_bipolar_weighted_00`, 6000 epochs | 0.731 | 1.0 | Oversampling `(0,0)` traded errors instead of producing a robust XOR landscape. |
| `metro_4hidden_full_bipolar_widefanout`, 6000 epochs | 0.610 | 0.75 | Wider fanout did not solve the basin problem. |
| `metro_full_bipolar_flipgrad`, 6000 epochs | 0.848 | 0.75 | Opposite-gradient plus full bipolar input was worse. |
| `metro_full_bipolar_zero_bias`, 6000 epochs | 0.825 | 0.75 | Removing trainable bias was not enough; `(0,0)` remained on the positive side. |
| `metro_full_bipolar_pattern_beta05`, 6000 epochs | 0.907 | 0.75 | Stronger physical output clamp saturated into a `not-11`-like rule. |
| `metro_full_bipolar_pattern_beta05_zero_bias`, 6000 epochs | 0.925 | 0.75 | Strong physical clamp plus zero bias still learned `not-11`, not XOR. |
| `metro_full_bipolar_pattern_hot`, 6000 epochs | 0.932 | 0.75 | Higher temperature did not escape the wrong false-case basin. |
| `metro_oldrecipe_false_output_prior`, quick probe | 1.531 | 0.75 | A false-output bias prior hurt the one-bit true cases, so it was not promoted. |

## Main Findings

- The current random-init local checkerboard setup still does **not** meet the
  target (`acc = 1.0` and MSE near `0.1`).
- The old sub-0.2 run is not reproduced by the same hyperparameters once
  random initialization is enforced. Its saved config shows it relied on
  `init_mode = :zero` for a discrete graph.
- Several failed runs have the same signature: output scores become exactly or
  nearly zero, and the contrastive gradient norm becomes zero. That means the
  plus/minus nudged states are often not producing a persistent differential
  response under these settings.
- The temperature/coupling normalization idea was useful diagnostically but
  not sufficient. It prevented immediate blow-up, but it also reduced margins.
- The opposite-gradient probe should **not** be treated as a sign correction.
  It produced one sub-0.5 stochastic result, but it did not produce a robust
  low-MSE solution and mostly stopped receiving gradient.
- The dominant new failure mode is now specific: the `(0,0)` input has no
  positive frozen evidence in the original protocol, and even the full-bipolar
  diagnostic tends to put the all-minus input in the same readout basin as the
  one-bit true cases. Stronger output clamping then saturates a `not-11`
  solution: `(11)` false is correct, but `(00)` false is wrong.
- Direct output-pattern clamping is cleaner than scalar readout clamping, but
  it is not sufficient. It makes the wrong basin sharper rather than producing
  the missing XOR separation.

## Next Ideas

- Treat the local checkerboard protocol as a representation problem, not just
  a sampler-tuning problem. The model repeatedly finds `not-11` or weak-margin
  rules.
- Test an architecture where the no-input/default state has a separate physical
  pathway from the one-bit patterns without using zero initialization.
- Add per-case response diagnostics before doing more long sweeps. The useful
  question is whether `(00)` gets a distinct plus/minus response at all.
- Keep Metropolis as the local baseline until it produces nonzero per-case
  contrastive responses under random init. Langevin tuning should follow that,
  not replace it.

## Simple Baseline Grid Follow-Up

Added:

```text
ext/IsingLearning/experiments/local_checkerboard_xor/simple_checkerboard_baseline_grid.jl
```

This is a smaller experiment-local grid for the simplest checkerboard task. It
uses the stabilized no-anneal worker path and searches side length, hidden size,
internal NN, internal scale, bias scale, temperature factor, inter-layer scale,
beta, and learning rate. It also supports experiment-local overrides for
Metropolis, BlockLangevin, and GlobalLangevin.

Additional tested probes:

| Probe | Best MSE | Best Acc | Interpretation |
|---|---:|---:|---|
| cold 2x2 smoke, low averaging | 0.547 | 1.0 | Not robust; disappeared with normal averaging. |
| same cold 2x2, `Minit=8`, eval 32 | 0.992 | 0.5 | Collapsed to zero readout/zero gradient. |
| 2x2, higher J/T, Metropolis | 0.857 | 1.0 | Correct signs transiently, but two arms remained at zero. |
| 4x4, NN=2, higher J/T, Metropolis | 0.968 | 0.75 | Larger layer and more internal connectivity still gave tiny margins. |
| 2x2, high initial bias scale | 0.985 | 0.5 | Bias created a default basin, then collapsed; did not solve XOR. |
| 2x2, GlobalLangevin, random continuous starts | 0.941 | 1.0 | Margins were microscopic and did not amplify. |
| 2x2, GlobalLangevin, continuous zero starts | 0.983 | 0.75 | Valid continuous control, still did not polarize. |
| 2x2, Metropolis, deterministic all-minus starts | 1.248 | 0.75 | Valid discrete background state; made a false default but killed true cases. |

The simple baseline grid reinforces the same conclusion: the current local
checkerboard protocol does not fail because one knob is slightly off. Random
starts collapse to zero readouts; deterministic low starts create a false
default but suppress the true cases; GlobalLangevin gives nonzero gradients but
does not amplify margins. The next change should probably alter the physical
encoding/architecture, not just continue scalar-readout hyperparameter search.
