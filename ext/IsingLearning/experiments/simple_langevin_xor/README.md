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

## Performance Note

The first version of this experiment accidentally serialized worker trajectories:
it started a worker and immediately waited for it. The current training loop
starts a batch of workers first and then collects them, so `workers > 1` now
actually parallelizes repeated initial states.
