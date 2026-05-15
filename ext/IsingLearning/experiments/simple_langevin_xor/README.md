# Simple LocalLangevin XOR

This folder contains the smallest controlled LocalLangevin XOR comparison:

```text
2 input spins -> 4 hidden spins -> 1 scalar output spin
```

The task uses direct bipolar XOR inputs:

- `(0, 0) -> [-1, -1]`
- `(0, 1) -> [-1, +1]`
- `(1, 0) -> [+1, -1]`
- `(1, 1) -> [+1, +1]`

The scalar target is `-1` for XOR false and `+1` for XOR true. The output
clamping is the masked built-in `Clamping`, so only the physical output spin is
nudged.

Both routes use:

- unadjusted `LocalLangevin`;
- continuous bounded states in `[-1, 1]`;
- trainable bilinear couplings and magnetic-field biases;
- no polynomial or double-well local potential;
- the normal Process/Composite machinery with `@repeat`, not manual `step!`.

## Routes

`normal`:

1. Free relaxation runs to the endpoint.
2. Plus/minus nudged phases restart from that endpoint.
3. `contrastive_gradient(free_endpoint, plus, minus, β)` is applied.

`split`:

1. Free relaxation runs for a short early window.
2. That early state is stored in `equilibrium_state`.
3. The free branch continues to the late endpoint.
4. Plus/minus nudged phases restart from the early state.
5. The contrastive gradient still uses the late free endpoint.

## First Quick Result

Run:

```text
ext/IsingLearning/experiments/simple_langevin_xor/runs/quick_20260511_202025
```

Settings: `T=0.02`, `stepsize=0.20`, `max_drift_fraction=0.60`,
`free=150`, `early=20`, `nudged=150`, `β=0.2`, `Minit=2`,
`eval_repeats=4`, `epochs=80`.

Observed:

- Normal route best logged region was about MSE `0.56`, accuracy `0.75`.
- Split route briefly reached MSE `0.437`, accuracy `1.0` at epoch 10, then
  degraded.

Interpretation: the composite and LocalLangevin plumbing work in both modes.
This setting is not yet stable enough to solve XOR, but the split route can
produce a better early signal than the normal route on this tiny graph.

## Normal Scalar-Output Follow-Up

The normal `2 -> 4 -> 1` route was tested with random initial states only.
Tried settings included:

- aggressive drift: `T=0.005..0.02`, `stepsize=0.5..1.0`, `β=1.0..1.5`,
  `free/nudged=900..1200`, `Minit=4`;
- no weight decay with `Minit=6`;
- long local-sweep relaxation with `free/nudged=7000`, about 1000 local sweeps
  on the seven-spin graph;
- colder starts down to `T=0.001` with larger initial weights.

Best scalar-output normal result so far was still partial: about MSE `0.63`
to `0.66`, with either `0.75` accuracy or brief `1.0` sign accuracy with weak
output magnitude. Increasing local sweeps did not solve it. Zero initialization
was tried only as a diagnostic and made the scalar route worse; it is not used
for the actual random-initial-state experiments.

Current interpretation: the scalar output is a poor proof target for
LocalLangevin. The dynamics can often learn the quadrant for most cases, but it
does not reliably drive the single output to a strong `-1/+1` XOR code.

## Two-Output LocalLangevin Control

To separate "LocalLangevin cannot learn" from "scalar output is awkward", a
two-output control was added:

```text
ext/IsingLearning/experiments/simple_langevin_xor/simple_2_16_2_locallangevin.jl
```

This uses the honest two bipolar input bits, a 16-spin hidden layer, and a
direct two-spin class code:

```text
false -> [ 1, -1]
true  -> [-1,  1]
```

It still uses random initial states, unadjusted `LocalLangevin`, masked direct
output clamping, no local potential, and the same plus/minus EP gradient.

Useful runs:

```text
runs/twoout_normal_20260511_220506
runs/twoout_normal_lowlr_20260511_220939
runs/twoout_normal_minit8_20260511_221408
runs/twoout_beta2_20260511_221840
```

Findings:

- `Minit=4`, `lr=0.0015`, `β=1.0`: reached MSE `0.3225`, accuracy `1.0`,
  then drifted to a worse basin.
- `Minit=4`, `lr=0.0007`, `β=1.0`: reached MSE `0.3170`, accuracy `1.0`,
  with slower drift.
- `Minit=8`, `lr=0.0007`, `β=1.0`: more stable; accuracy stayed `1.0` after
  epoch 200 and MSE stayed around `0.35..0.40`.
- `β=2.0` with lower LR was worse; it weakened the outputs instead of
  improving saturation.

This is the current LocalLangevin evidence: the normal EP sign and random-init
statistical averaging do produce learning with LocalLangevin, but the continuous
two-output run is still not below the desired MSE `0.1`. The next knob to tune
is not zero initialization; it is output saturation under random initial-state
statistics, likely through temperature/stepsize/relaxation and possibly better
early stopping or optimizer scheduling.

## Stable LocalLangevin Retest

After the Langevin sampler was made more stable near the bounds, the two-output
control was rerun with the normal route, no split snapshot, and no local
potential:

```text
runs/stable_langevin_2_16_2_minit1_long_20260514
```

Settings:

| parameter | value |
|---|---:|
| hidden units | 16 |
| output units | 2 |
| temperature | 0.07 |
| stepsize | 0.8 |
| beta | 1.0 |
| learning rate | 0.0006 |
| free relaxation | 1200 local updates |
| nudged relaxation | 1200 local updates |
| random starts per training sample | 1 |
| validation repeats | 24 |
| epochs | 3000 |

Result:

| epoch | MSE | accuracy |
|---:|---:|---:|
| 0 | 1.030516 | 0.5 |
| 300 | 0.385842 | 1.0 |
| 600 | 0.245457 | 1.0 |
| 2400 | 0.229119 | 1.0 |
| 2700 | 0.175700 | 1.0 |
| 3000 | 0.125345 | 1.0 |

Final mean outputs:

```text
(0,0): [ 0.642, -0.655]
(0,1): [-0.634,  0.595]
(1,0): [-0.602,  0.641]
(1,1): [ 0.689, -0.727]
```

This is the best current LocalLangevin result in this folder. The important
change relative to the older averaged runs was not more random-start averaging;
the `Minit=1` control worked better here. The likely reason is that averaging
several stochastic trajectories made the gradient less coherent for this small
system, while the validation repeat average still measures robustness after
training.

## Scalar `2 -> 4 -> 1` Curriculum

The scalar-output task finally worked from random initialization with a target
curriculum:

```text
simple_2_4_1_curriculum.jl
runs/analyticpath_random_curriculum_20260514
```

Architecture:

```text
2 input spins -> 4 hidden spins -> 1 scalar output spin
```

The successful run used random initial weights, no local potential, unadjusted
`LocalLangevin`, and direct masked scalar output clamping. The target was not
set to `±1` immediately. Instead:

| stage | target scale | epochs | learning rate |
|---:|---:|---:|---:|
| 1 | 0.25 | 1200 | 0.0010 |
| 2 | 0.50 | 1200 | 0.0008 |
| 3 | 1.00 | 2400 | 0.0005 |

Other settings:

| parameter | value |
|---|---:|
| temperature | 0.07 |
| stepsize | 0.8 |
| beta | 1.0 |
| free relaxation | 1200 |
| nudged relaxation | 1200 |
| Minit | 1 |
| validation repeats | 24 |
| initial weight scale | 0.12 |

Results against the full `±1` scalar target:

| epoch | MSE | accuracy | mean outputs |
|---:|---:|---:|---|
| 0 | 1.341617 | 0.5 | `[-0.907, -0.990, 0.810, 0.167]` |
| 600 | 0.560708 | 1.0 | `[-0.128, 0.850, 0.181, -0.112]` |
| 1500 | 0.217383 | 1.0 | `[-0.194, 0.770, 0.596, -0.944]` |
| 2400 | 0.118574 | 1.0 | `[-0.403, 0.813, 0.774, -0.822]` |
| 3000 | 0.064740 | 1.0 | `[-0.776, 0.577, 0.827, -0.988]` |
| 3600 | 0.030847 | 1.0 | `[-0.661, 0.990, 0.909, -0.989]` |

What made this work:

- Direct full-strength scalar clamping made the output saturate early into the
  wrong three-positive attractor, after which the gradient became very small.
- A weak initial target kept the scalar output responsive long enough to learn
  the XOR sign structure.
- Once the signs were correct, increasing the target scale to `1.0` improved
  the output magnitudes instead of trapping the system in the wrong basin.
- `Minit=1` again worked better than averaging several random-start gradients
  for this tiny system; validation still averages repeated starts.

The analytic corner-detector control was also checked:

```text
runs/analytic_2_4_1_stablecheck_20260514
```

It validates at MSE `0.000107`, accuracy `1.0`, with means
`[-0.989, 0.991, 0.989, -0.990]`. So the architecture can represent scalar XOR;
the curriculum is solving the optimization problem from random initialization.

### Best Checkpoint Restore

The scalar run is stochastic after the signs are learned. It can reach a good
solution and then drift to a worse validation MSE even though accuracy stays
`1.0`. The curriculum script now stores the best validation parameters and
restores them at the end.

Confirmed run:

```text
runs/curriculum_bestrestore_20260514
```

| event | epoch | MSE | accuracy | mean outputs |
|---|---:|---:|---:|---|
| best logged checkpoint | 4500 | 0.038957 | 1.0 | `[-0.741, 0.830, 0.939, -0.763]` |
| restored best re-eval | 4500 | 0.027362 | 1.0 | `[-0.742, 0.910, 0.989, -0.815]` |

The re-evaluated MSE differs from the logged MSE because validation is still a
finite stochastic average. The important point is that the restored parameters
remain in the low-MSE, correct-sign basin.

### Full Target With Smaller Beta

The curriculum is not strictly necessary. A direct full-`±1` target works if
the clamping strength is reduced. This matters because `target_scale = 0.25`
and `β = 0.25` are not mathematically identical for the direct clamping term:

```text
H_clamp = β/2 * (s - y)^2
        = β/2 * s^2 - β*y*s + constant
```

Reducing `y` weakens the linear target field `β*y*s` but keeps the clamping
curvature `β/2 * s^2`. Reducing `β` weakens both the target field and the
curvature. Even so, the experiment shows that the practical issue was mostly
too-strong clamping.

Run folder:

```text
runs/fulltarget_beta_grid_20260514
```

Shared settings:

| parameter | value |
|---|---:|
| target scale | 1.0 |
| temperature | 0.07 |
| stepsize | 0.8 |
| learning rate | 0.0006 |
| free/nudged relaxation | 1200 / 1200 |
| Minit | 1 |
| validation repeats | 24 |
| epochs | 3000 |

Results:

| beta | best MSE | best accuracy | final MSE | final accuracy | best epoch |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 1.093429 | 0.75 | 1.314658 | 0.5 | 300 |
| 0.2 | 0.000085 | 1.0 | 0.031067 | 1.0 | 2400 |
| 0.35 | 0.046558 | 1.0 | 0.046558 | 1.0 | 3000 |
| 0.5 | 0.986385 | 0.75 | 0.986385 | 0.75 | 3000 |
| 0.75 | 0.105123 | 1.0 | 0.113686 | 1.0 | 2700 |

Interpretation:

- `β = 0.2` is the cleanest direct full-target run so far.
- `β = 0.1` is too weak to reliably push the system into the correct basin.
- `β = 0.5` and above often recreate the old failure mode: the scalar output
  saturates into the wrong branch and the gradient becomes too small or
  unhelpful.
- The target curriculum worked because it effectively avoided the same
  too-strong early nudging regime. It is a stabilization trick, not a
  representational requirement.

### Nudged Annealing Note

The idea was to briefly increase temperature during the nudged branches so the
target perturbation can escape a too-strong free attractor. This is physically
reasonable, but the implementation has to preserve the same Langevin context as
the free phase. A version with separate plus/minus sampler contexts failed its
flat-temperature control because the Hamiltonian cache was not guaranteed to
match the restored free state after `setgraph!`.

Current conclusion:

- The attractor-depth concern is real.
- Target-scale curriculum plus best-checkpoint restore is already enough for
  scalar `2 -> 4 -> 1`.
- Nudged annealing should only be trusted after the no-anneal control exactly
  reproduces the standard curriculum path.

## Performance Note

The first version of this experiment accidentally serialized worker trajectories:
it started a worker and immediately waited for it. The current training loop
starts a batch of workers first and then collects them, so `workers > 1` now
actually parallelizes repeated initial states.
