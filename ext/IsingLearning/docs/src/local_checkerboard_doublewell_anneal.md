# Local Checkerboard Double-Well Anneal

This note records the first test of adding a static double-well potential to
the local checkerboard XOR experiments.

Generated run folders for failed local-checkerboard probes were compacted into:

```text
ext/IsingLearning/experiments/local_checkerboard_xor/runs/_summary/
```

That folder contains a compact CSV, a markdown archive, and one comparison PNG.
The old high-MSE run directories were removed after their parameters and
results were recorded there.

The follow-up stabilized search is summarized separately in:

```text
ext/IsingLearning/experiments/local_checkerboard_xor/runs/_summary/local_checkerboard_stabilized_search.md
```

Main update from that pass: the old sub-0.2 Metropolis run depended on a
discrete zero-init recipe. With random initialization enforced, no tested
Metropolis/Langevin-adjacent stabilization reached the target MSE. The dominant
failure mode is still weak or zero plus/minus response, visible as tiny readout
margins and zero contrastive gradients.

Follow-up update: full-bipolar input diagnostics, stronger physical output
clamping, zero-bias projection, wider fanout, and a false-output bias prior did
not solve the local checkerboard task. The dominant failure is now sharper:
the `(0,0)` case falls into the same positive/readout-true basin as one-bit
inputs, while `(1,1)` is often correctly classified as false. Stronger output
clamping tends to harden this into a `not-11` rule rather than XOR.

Second follow-up: a focused simple checkerboard baseline grid was added in
`simple_checkerboard_baseline_grid.jl`. It tested the plain 2x2 and 4x4 local
checkerboard systems with in-layer connections, stronger couplings, larger
internal NN, larger bias scale, GlobalLangevin controls, continuous zero-start
controls, and valid all-minus discrete starts. None reached a useful low-MSE
solution. The best transient sign-correct run was still high-MSE (`~0.86`) and
had zero-valued arms. This points away from more scalar-readout knob tuning and
toward changing the physical encoding or architecture.

Third follow-up: the scalar-readout path was not enough, but a structured local
2x2 checkerboard circuit with an explicit hidden `A`, `B`, and `AND` route now
works. See `local_checkerboard_structured_success.md`. The best short-training
run reached MSE `0.095703125` and accuracy `1.0`, with a saved graph under
`runs/structured_and_seed_20260511_052236/`. The key lesson is architectural:
the local system can run XOR, but the previous random shallow local graph was
not reliably discovering the AND feature and readout amplification by itself.

## Implementation

The experiment file now has an optional static local potential:

```math
V(s) = a(s^4 - 2s^2)
```

The minima are at `s = +/-1`; the barrier from either minimum to `s = 0` is
`a`. This is implemented as `StaticDoubleWell` inside the local checkerboard
experiment code, not as `Quadratic + Quartic`, because `Quadratic` is currently
treated by the learning code as a trainable local-potential parameter.

The sampler can also be wrapped in `TemperatureAnnealedSampler`, which writes a
power-law scheduled temperature to the graph before each underlying sampler
step. The wrapped sampler itself is unchanged.

## Tested Schedule

The intended test was:

- free phase: weak anneal;
- nudged phase: start at a temperature on the order of the double-well barrier,
  then cool below it;
- average over `Minit = 4` random starts.

Two 2x2 checkerboard runs were tested.

### Mild Run

```text
run: ext/IsingLearning/experiments/local_checkerboard_xor/runs/doublewell_annealed_langevin_20260510_162730/
barrier = 0.02
free T = 0.03 -> 0.005
nudged T = 0.02 -> 0.00125
BlockLangevin stepsize = 0.05
beta = 0.05
inter scale = 0.05
free/nudged relaxation = 500 / 500
Minit = 4
```

Result: MSE moved only from `1.118` to `1.033`; accuracy ended at `0.5`.

### Stronger Run

```text
run: ext/IsingLearning/experiments/local_checkerboard_xor/runs/doublewell_annealed_langevin_stronger_20260510_163330/
barrier = 0.03
free T = 0.045 -> 0.00333
nudged T = 0.03 -> 0.00167
BlockLangevin stepsize = 0.08
beta = 0.2
inter scale = 0.10
internal scales = 0.01
free/nudged relaxation = 500 / 500
Minit = 4
```

Result: MSE moved from `1.127` to `1.020`; accuracy ended at `0.5`.

## Interpretation

The double well and annealing mechanism is technically working, but this first
version did not fix the learning bottleneck. In both runs the readout scores
stayed small and mostly same-signed. That means the issue is not simply that
the continuous spins need a local double well to reach `+/-1`; the scalar
checkerboard readout still does not get a strong class-separated response.

The next useful tests should change one thing at a time:

- direct output-pattern clamping instead of scalar readout clamping;
- stronger or normalized layer-to-layer coupling while checking symmetry;
- GlobalLangevin for free minimization, with the same static double well;
- longer training only after margins start separating in short probes.

## Output Pattern Clamping Probe

The old `output_clamp_mode = :pattern` path used the generic full-graph
`InteractiveIsing.Clamping`. That is not the right local test here, because the
generic clamping term has one scalar `beta` and a full-state target vector. If
only the output target is filled, the remaining graph entries are target `0`
and still feel the clamping force.

The experiment now uses an output-only term:

```math
H_\mathrm{out}(s) =
\frac{\beta}{2}\sum_k (s_{i_k} - y_k)^2
```

where `i_k` are exactly `checker_output_idxs`. Non-output spins get zero force
from this term. The plus/minus phases set the same physical output pattern with
opposite `beta` signs, and the free phase resets this term to `beta = 0`.

First direct comparison:

```text
run: ext/IsingLearning/experiments/local_checkerboard_xor/runs/pattern_clamp_doublewell_langevin_20260510_192733/
barrier = 0.03
free T = 0.045 -> 0.00333
nudged T = 0.03 -> 0.00167
BlockLangevin stepsize = 0.08
beta = 0.2
inter scale = 0.10
internal scales = 0.01
free/nudged relaxation = 500 / 500
Minit = 4
output clamp = pattern
```

Result: the run did not solve the 2x2 local checkerboard task. It started at
MSE `1.127`, accuracy `0.5`, and ended at MSE `1.171`, accuracy `0.25`. The
gradient was nonzero, so the output-pattern term is active, but the readout
scores remained small and did not separate by XOR class.

Conclusion: the scalar readout clamp was not the only bottleneck. Directly
nudging the physical output pattern is cleaner and should remain available, but
this architecture/dynamics setting still needs a stronger way to create class
separation.

## No Double-Well Control

The same output-pattern clamping path was also tested with the double well
disabled:

```text
run: ext/IsingLearning/experiments/local_checkerboard_xor/runs/pattern_clamp_nodw_langevin_20260510_201245/
barrier = 0
constant T = 0.01
BlockLangevin stepsize = 0.08
beta = 0.2
inter scale = 0.10
internal scales = 0.01
free/nudged relaxation = 500 / 500
Minit = 4
output clamp = pattern
```

Result: this also did not solve the 2x2 local checkerboard task. It started at
MSE `1.136`, accuracy `0.5`, and ended at MSE `1.257`, accuracy `0.5`. The
readout scores drifted mostly positive, so the failure is not caused by the
static double-well potential.

For now, ignore the double well in local checkerboard experiments. The more
important axes are coupling scale/normalization, temperature relative to the
interaction field, and whether the local architecture can create a robust XOR
separation with these partially frozen input patterns.

## No-Double-Well Grid Search

A focused no-double-well grid was run with direct output-pattern clamping,
symmetric weights, random initialization, and the 2x2 local checkerboard code.
This was only a screening pass, using short `50/50` relaxations, `Minit = 1`,
and 20 epochs.

```text
run: ext/IsingLearning/experiments/local_checkerboard_xor/runs/grid_screen_continuous_nodw_pattern_20260510_204557/
configs: 9 BlockLangevin + 3 GlobalLangevin
output clamp = pattern
doublewell barrier = 0
```

Best screen result:

```text
BlockLangevin
T = 0.01
stepsize = 0.08
beta = 0.2
inter-layer scale = 0.25
best MSE = 0.650
best accuracy = 1.0
```

This was not robust. A longer rerun with the same hyperparameters but default
seeds returned to random-level behavior. A seed-matched longer rerun preserved
some sign separation, but MSE stayed high:

```text
run: ext/IsingLearning/experiments/local_checkerboard_xor/runs/best_screen_longer_seedmatched_nodw_pattern_20260510_210233/
free/nudged relaxation = 250 / 250
Minit = 4
eval repeats = 16
start MSE = 0.810, accuracy = 1.0
final MSE = 0.838, accuracy = 0.75
```

A second stronger-coupling grid scanned `inter-layer scale = 0.5` and `1.0`:

```text
run: ext/IsingLearning/experiments/local_checkerboard_xor/runs/grid_strongJ_nodw_pattern_20260510_210918/
best sign-correct MSE = 0.767
```

Removing weight decay and raising the learning rate also did not solve it:

```text
run: ext/IsingLearning/experiments/local_checkerboard_xor/runs/strongJ_nodecay_lr001_nodw_pattern_20260510_211634/
J = 1.0
T = 0.01
stepsize = 0.12
beta = 0.5
lr = 0.01
weight_decay = 0
final MSE = 0.935
```

Interpretation: the local checkerboard setup can sometimes get the four XOR
signs correct, but the output amplitudes stay small. The failure mode is no
longer "wrong clamping" or "missing double well"; it is that the current local
architecture/dynamics/gradient combination does not reliably amplify the
physical output pattern toward `+/-1`.

## Same-Layer NN and Inter-Layer Radius Scan

The first no-double-well screens did not scan both locality knobs. The grid
file now varies:

- `internal_nn`: same-layer nearest-neighbor range inside input, hidden, and
  output layers;
- `inter_radius`: radius-limited layer-to-layer fanout.

Two short screens were run with direct output-pattern clamping, symmetric
weights, random initialization, `Minit = 1`, and `25/25` free/nudged
relaxations. These are cheap sign screens only, not robustness proofs.

Baseline coupling:

```text
run: ext/IsingLearning/experiments/local_checkerboard_xor/runs/grid_NN_radius_fast_20260510_214619/
T = 0.01
stepsize = 0.08
beta = 0.2
inter-layer scale = 0.25
```

Best short result:

```text
internal_nn = 2
inter_radius = 1.01
best MSE = 0.674
best accuracy = 1.0
```

Stronger clamp/coupling:

```text
run: ext/IsingLearning/experiments/local_checkerboard_xor/runs/grid_NN_radius_strong_fast_20260510_215621/
T = 0.01
stepsize = 0.12
beta = 0.5
inter-layer scale = 0.5
```

Best short result:

```text
internal_nn = 3
inter_radius = sqrt(2)
best MSE = 0.846
best accuracy = 1.0
```

The baseline `internal_nn = 2, inter_radius = 1.01` candidate did not survive
a more averaged rerun:

```text
run: ext/IsingLearning/experiments/local_checkerboard_xor/runs/best_NN2_r1_longer_nodw_pattern_20260510_220611/
Minit = 4
eval repeats = 16
free/nudged relaxation = 150 / 150
best MSE = 0.966
best accuracy = 0.75
```

Interpretation: both nearest-neighbor settings matter, but the current wins are
not robust. Increasing inter-layer radius is not automatically better; it also
changes the local field scale and can wash out the locality of the code. The
next scans should normalize coupling strength by fanout or degree before
comparing radii.

The saved best graphs from the short screen and the averaged rerun were checked
for adjacency symmetry after training. Both had max `|J_ij - J_ji| = 0.0`, so
the current optimizer path did not break the energy-model symmetry in these
runs.

An unadjusted GlobalLangevin control was also run over the same
`internal_nn x inter_radius` grid:

```text
run: ext/IsingLearning/experiments/local_checkerboard_xor/runs/grid_global_NN_radius_fast_20260510_222130/
GlobalLangevin(adjusted = false)
T = 0.01
stepsize = 0.03
beta = 0.2
inter-layer scale = 0.25
Minit = 1
free/nudged relaxation = 25 / 25
```

Best short result:

```text
internal_nn = 1
inter_radius = sqrt(2)
best MSE = 0.778
best accuracy = 0.75
```

So, in this corrected topology scan, global Langevin did not beat BlockLangevin.
It may still be worth scanning global `stepsize` and temperature more
aggressively, but just switching from block to global updates is not enough.

## Short Free, Longer Nudged Probe

An asymmetric relaxation screen was run to test whether the free phase can be
short while the nudged phase gets more time to propagate the output
perturbation backward.

```text
run: ext/IsingLearning/experiments/local_checkerboard_xor/runs/grid_free_short_nudged_long_20260511_002252/
output clamp = pattern
doublewell barrier = 0
Minit = 2
eval repeats = 8
epochs = 60
```

Best result:

```text
topology = baseline NN2/r1
free/nudged relaxation = 25 / 150
T = 0.01
stepsize = 0.08
beta = 0.2
inter-layer scale = 0.25
best MSE = 0.813
best accuracy = 1.0
```

This was only a small improvement over the `25/75` and `50/300` variants.
Longer nudged phases increased the gradient norm but did not create strong
output amplitudes. The stronger topology (`NN3/r141`, stronger clamp/coupling)
did not benefit from the asymmetric schedule in this screen.

Interpretation: short-free/long-nudged is plausible and not harmful for the
baseline topology, but it does not by itself solve the weak-output problem.

A focused rerun of the doubled schedule was also checked:

```text
run: ext/IsingLearning/experiments/local_checkerboard_xor/runs/best_NN2_r1_free50_nudged300_focused_20260511_010022/
free/nudged relaxation = 50 / 300
Minit = 4
eval repeats = 16
epochs = 120
best MSE = 0.920
best accuracy = 1.0
```

This is worse than the cheap `25/150` screen and still has very weak output
scores. Takeaway: simply increasing both free and nudged relaxation does not
fix the local checkerboard setup.

## Next Useful Things To Try

- Keep the symmetry diagnostic after each training run. The checked runs stayed
  symmetric, but this should remain a hard acceptance check for local learning.
- Normalize inter-layer and same-layer coupling scales by degree, e.g.
  `1/sqrt(fanout)` or `1/fanout`, so `inter_radius` and `internal_nn` can be
  compared at a fixed local field scale.
- Split learning rates or trainability for internal versus inter-layer weights.
  Random recurrent layer weights appear to add noise unless they are initialized
  very small or controlled separately.
- Run the same 2x2 setup with true discrete Metropolis after avoiding the
  experiment-local temperature wrapper issue. If discrete dynamics works while
  continuous Langevin fails, the bottleneck is the continuous relaxation scale.
- Track physical output-pattern MSE in addition to scalar checkerboard readout
  MSE. A sign-correct scalar readout can hide weak or malformed output states.
- After degree normalization, repeat the search over `T`, `stepsize`, `beta`,
  `free_relaxation`, and `nudged_relaxation`. Temperature should be interpreted
  relative to the maximum local interaction field, not as an absolute number.
