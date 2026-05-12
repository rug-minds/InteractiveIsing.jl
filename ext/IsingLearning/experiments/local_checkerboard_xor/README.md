# Local Checkerboard XOR

This folder contains the current local-pattern XOR experiment:

- main file: `local_checkerboard_xor.jl`
- run outputs: `runs/local_checkerboard_xor_<timestamp>/`

## Encoding

The input is deliberately not a four-case one-hot code. There are only two input
bits and two physical checkerboard masks in the input layer:

- bit A = 1 freezes the white sites of checkerboard A to `+1`;
- bit B = 1 freezes the white sites of checkerboard B to `+1`;
- `(0, 0)` freezes no input spins;
- `(1, 0)` freezes checkerboard A;
- `(0, 1)` freezes checkerboard B;
- `(1, 1)` freezes both masks, i.e. the whole input code.

The input layer itself remains internally connected. For embedded/inlaid codes,
non-code input spins are never frozen by the input bits and keep participating
in the dynamics.

The output is a scalar linear checkerboard readout on the final layer:

```julia
score = dot(readout, output_code_state)
```

with target `-1` for XOR false and `+1` for XOR true. This uses
`LinearReadoutClamping`, so the supervised cost is directly on a linear output
readout rather than on a hidden one-hot class variable.

## Implemented Tests

`local_checkerboard_xor.jl` defines these configs:

- `checker_2x2_global`: `2x2 -> 2x2 -> 2x2`, full checkerboard code.
- `checker_4x4_global`: `4x4 -> 4x4 -> 4x4`, full checkerboard code.
- `checker_8x8_global4`: `8x8 -> 8x8 -> 8x8`, full-layer checkerboard code.
- `checker_8x8_inlaid4`: `8x8 -> 8x8 -> 8x8`, `4x4` checkerboard code inlaid with stride 2.

Each layer has internal local connections. Adjacent layers use radius-limited
signed random connections. The graph uses a custom `CheckerboardInputIndexSet`
that precomputes the active sampler list for all four input-bit combinations.

## How To Run

Quick smoke:

```bash
ISING_LOCAL_XOR_CONFIGS=checker_2x2_global \
ISING_LOCAL_XOR_EPOCHS=2 \
ISING_LOCAL_XOR_LOG_EVERY=1 \
ISING_LOCAL_XOR_MINIT=1 \
ISING_LOCAL_XOR_EVAL_REPEATS=2 \
julia --project=ext/IsingLearning \
  ext/IsingLearning/experiments/local_checkerboard_xor/local_checkerboard_xor.jl
```

Baseline 2x2 and 4x4:

```bash
ISING_LOCAL_XOR_CONFIGS=checker_2x2_global,checker_4x4_global \
ISING_LOCAL_XOR_EPOCHS=200 \
ISING_LOCAL_XOR_LOG_EVERY=20 \
ISING_LOCAL_XOR_MINIT=4 \
ISING_LOCAL_XOR_EVAL_REPEATS=8 \
ISING_LOCAL_XOR_THREADS=4 \
julia --project=ext/IsingLearning \
  ext/IsingLearning/experiments/local_checkerboard_xor/local_checkerboard_xor.jl
```

Stronger-coupled 2x2 probe:

```bash
ISING_LOCAL_XOR_CONFIGS=checker_2x2_global \
ISING_LOCAL_XOR_EPOCHS=500 \
ISING_LOCAL_XOR_LOG_EVERY=50 \
ISING_LOCAL_XOR_MINIT=4 \
ISING_LOCAL_XOR_EVAL_REPEATS=8 \
ISING_LOCAL_XOR_THREADS=4 \
ISING_LOCAL_XOR_INTER_WEIGHT_SCALE=0.25 \
ISING_LOCAL_XOR_INPUT_INTERNAL_SCALE=0.1 \
ISING_LOCAL_XOR_HIDDEN_INTERNAL_SCALE=0.1 \
ISING_LOCAL_XOR_OUTPUT_INTERNAL_SCALE=0.1 \
ISING_LOCAL_XOR_BETA=0.2 \
ISING_LOCAL_XOR_LR=0.01 \
ISING_LOCAL_XOR_TEMP=0.01 \
julia --project=ext/IsingLearning \
  ext/IsingLearning/experiments/local_checkerboard_xor/local_checkerboard_xor.jl
```

## Current Results

### Structured Local AND Seed

A structurally seeded 2x2 local checkerboard run now reaches the target
validation range:

- file: `structured_and_seed_search.jl`
- calibration run: `runs/structured_and_seed_20260511_051030/`
- short training run: `runs/structured_and_seed_20260511_052236/`
- best saved graph:
  `runs/structured_and_seed_20260511_052236/structured_2x2_T0.001_ib2.0_f3.5_o0.8_ob0.4_b0.1_lr0.002/structured_2x2_T0.001_ib2.0_f3.5_o0.8_ob0.4_b0.1_lr0.002_best_graph.jld2`

Best short-training result:

```text
accuracy = 1.0
best MSE = 0.095703125
scores = [-2.0, 1.0, 1.625, -1.375] at the best logged epoch
```

What made this work:

- inactive input bits are still not frozen, but the input-code sites have a
  fixed negative magnetic field, so absent bits relax toward `-1`;
- the hidden layer is seeded by writing explicit weights/biases for local `A`,
  `B`, and `AND` feature routes;
- the output uses two local readout channels, one over each output checkerboard
  mask, instead of forcing all supervision through one scalar difference mode;
- the run uses a valid discrete all-minus background, not zero init;
- temperature is very low (`T = 0.001`) relative to the seeded interaction
  scale, so the seeded local circuit acts as a stable attractor.

This is not yet proof that random local checkerboard graphs learn the feature
from scratch. It shows the local checkerboard architecture can represent and
stably run XOR when it has an explicit local `AND` pathway. The previous random
local searches likely failed because they were asking a shallow local graph to
discover that pathway and amplify a scalar readout at the same time.

Concretely, this "seed" is not a hidden-state initialization. The hidden states
still start from the configured all-minus background and relax normally. The
seed is a set of initial graph parameters written by
`apply_structured_and_seed!`:

- choose hidden spins `hA`, `hB`, and `hAND` from the hidden `2x2`
  checkerboard masks;
- connect input-A sites to `hA` and `hAND` with positive symmetric weights
  `f / length(input_A)`;
- connect input-B sites to `hB` and `hAND` with positive symmetric weights
  `f / length(input_B)`;
- add a negative magnetic field `b[hAND] -= f`, so `hAND` only turns on
  robustly when both input masks are active;
- connect the true output mask as `+hA + hB - 2hAND`;
- connect the false output mask as `-hA - hB + 2hAND`;
- add a small false-default output bias.

The successful run used `f = 3.5`, output scale `o = 0.8`, and output bias
`0.4`. Since `E_J = -1/2 sum_ij J_ij s_i s_j`, positive weights align spins,
and the hidden/output wiring implements the XOR combination `A + B - 2AB`.

The local checkerboard setup now learns the XOR signs in the small `2x2`
Metropolis case, but it has not yet reached the desired low readout MSE. The
remaining problem is readout polarization: one or two cases stay too close to
zero even when the signs are correct.

Current best true-symmetric runs:

- `runs/metropolis_2x2_sym_lazy_stronger_20260509_234023/`
  - true symmetric in-layer weight generation;
  - `state=discrete`, `dynamics=Metropolis`, `T=0.01`, `β=0.2`,
    `free/nudged=150/150`, `Minit=8`, `lr=0.005`, no weight decay;
  - best MSE `0.259880`, accuracy `1.0`;
  - best scores `[-0.824, +0.633, +0.402, -0.281]`.
- `runs/metropolis_2x2_sym_lazy_T0010_inter010_20260510_000037/`
  - same recipe but `inter_weight_scale=0.10`, `lr=0.003`;
  - best MSE `0.245426`, accuracy `1.0`;
  - best scores `[-0.430, +0.426, +0.898, -0.438]`.
- `runs/metropolis_2x2_sym_lazy_seed5_T0010_inter010_20260510_001506/`
  - seed sweep with the same hyperparameters;
  - best MSE `0.266678`, accuracy `1.0`;
  - best scores `[-0.813, +0.391, +0.406, -0.445]`.

Strict no-zero Metropolis rerun:

- `runs/all_metropolis_random_symmetric_20260510_015004/`
  - configs: `checker_2x2_global`, `checker_4x4_global`,
    `checker_8x8_global4`, `checker_8x8_inlaid4`;
  - all configs used `state=discrete`, `dynamics=Metropolis`,
    `init_mode=random`, and symmetric adjacencies with `maximum(abs(J-J')) = 0`;
  - discrete `init_mode=zero` is now rejected by the experiment setup;
  - none of the four configs reached a useful low-MSE classifier from random
    initial states. The best transient results were `checker_4x4_global`
    MSE `0.899338`, accuracy `1.0`, and `checker_8x8_inlaid4` MSE `0.926159`,
    accuracy `1.0`, but both had margins of only `0.00390625`;
  - final MSE values stayed near `1.0`: `2x2=1.055176`, `4x4=0.951138`,
    `8x8_global=1.004637`, `8x8_inlaid=0.969120`.

Bigger-graph Metropolis reruns:

- `runs/bigger_metropolis_random_symmetric_20260510_020355/`
  - tested `12x12` global checkerboard, `16x16` global checkerboard, and
    `16x16` with an inlaid `8x8` checkerboard;
  - increased internal connectivity to `NN=2` or `NN=3` and increased
    inter-layer fanout radius to `2.05`, `2.55`, or `3.05`;
  - used `state=discrete`, `dynamics=Metropolis`, `init_mode=random`,
    `temp_is_factor=true`, and symmetric adjacency assertions;
  - best MSE values remained near `1.0`: `12x12=0.986141`,
    `16x16_global=0.982389`, `16x16_inlaid=0.989913`.
- `runs/bigger_strong_metropolis_random_symmetric_20260510_021146/`
  - repeated the larger test with stronger inter-layer weights, stronger
    output-layer internal coupling, larger fanout, and higher gradient clip;
  - best observed case was `checker_12x12_global12_strong_nn3_r35` with
    MSE `0.965419`, accuracy `1.0`, but the scores were still tiny:
    `[-0.0260, +0.0148, +0.0069, -0.0221]`;
  - conclusion: simply increasing graph size, checkerboard size, local
    connectivity, and inter-layer fanout does not solve the scalar-readout
    polarization problem under random discrete initialization.

Runs that are now considered diagnostic only:

- `runs/metropolis_2x2_symmetric_nodecay_20260509_213406/`
  reached MSE `0.183666`, accuracy `1.0`, but it used post-hoc adjacency
  symmetrization after independently sampling both directed edge weights. That
  gives a different initialization distribution from the current true
  symmetric generator, so it is not the preferred baseline.
- `runs/metropolis_2x2_sym_lazy_randominit_T0010_inter010_20260510_001904/`
  used valid random discrete initial states. It collapsed to weak responses and
  MSE around `0.93`. For this tiny local system, midpoint/quench starts produce
  a much stronger learning signal than random starts.
- `runs/langevin_2x2_sym_lazy_zero_T0010_inter010_20260510_002224/`
  used continuous Langevin with valid zero starts. It did not solve XOR:
  best MSE was about `0.836`, accuracy `0.75`. With these hyperparameters,
  Langevin moves the scores slowly and does not polarize the false/true arms.

## Implementation Notes

Two reproducibility issues were fixed while tuning this experiment:

- `BlockLangevin` now respects a context RNG override via
  `Input(:dynamics, rng=...)`. Before that, it always constructed its own
  `MersenneTwister()` in `init`, so shell-level `Random.seed!` did not control
  sampler paths.
- Worker gradient buffers are now accumulated immediately after each finished
  `(sample, init)` job. This matters when `Minit > workers`; a reused worker
  must not overwrite a previous contrastive buffer before it has been added to
  the batch gradient.
- Local checkerboard graphs now use true symmetric in-layer weight generation.
  The generator samples each undirected edge once and inserts both directions
  with the same value. The experiment still asserts `J_ij == J_ji` after graph
  construction.

## Interpretation

This is not the same difficulty as the earlier all-to-all two-input success.
Here `(0, 0)` means no clamped input at all, and `(1, 1)` means the entire input
code is clamped to the same sign. With the current shallow local graph and scalar
checkerboard output readout, the optimizer can find the XOR signs, but it has
not yet formed strong false/true attractors for all four cases.

The best current window for the `2x2` local task is:

```text
dynamics = Metropolis
state = discrete
T = 0.01
free/nudged relaxation = 150/150
Minit = 8
β = 0.2
lr = 0.003
weight_decay = 0
inter_weight_scale = 0.10
internal scales = 0.014142...
```

This finds the correct XOR signs, but not yet target-strength readouts. The
remaining concrete issue is not classification; it is increasing the weak arms
toward `±1` without losing the other cases.

Promising next tests:

- Try a two-channel output readout instead of one scalar checkerboard readout.
  A scalar readout asks one mode to separate all four cases; a two-channel
  readout may stabilize false/true attractors more cleanly while still using
  local patterns.
- Add a second hidden layer or a slightly larger hidden layer for the local
  graph. The false cases are `(0,0)` and `(1,1)`, which are physically very
  different inputs but share the same label; a single shallow local hidden layer
  may not reliably form that invariant.
- Tune output-layer internal coupling separately from input/hidden internal
  coupling. The current failures are mostly output polarization failures.
- Revisit Langevin with a dedicated temperature/stepsize sweep. The first
  direct transfer of the Metropolis recipe to continuous Langevin was poor.
