# R8 Rescue Series

Created: 2026-06-04

Goal: get the single-hidden-local `r=8` architecture to show sustained learning
from scratch. Do not resume from checkpoints in this series unless explicitly
testing checkpoint initialization.

## Current Hypotheses

- The flat `0.10` collapse is likely a deterministic single-class readout
  attractor, not ordinary noisy non-learning. When output replicas all settle
  into the same class pattern, balanced test accuracy stays exactly near chance.
- Strong nudging plus clipping can create too hard an output clamp. Lower beta
  should test whether r8 collapses because the target phase overwhelms the free
  phase.
- The r8 graph may be too dense for the default Adam step. Lower learning rates
  and larger minibatches should test whether the failure is optimizer instability.
- If Metropolis remains flat, the next branch should compare local Langevin and
  global Langevin, then repeat the best recipe across several random seeds and
  average the curves.

## Manager Path

This series uses chunked sample-index jobs. Each worker receives sample indices,
loads columns from shared `x`/`y` matrices into its own buffers, then runs the
normal process inline inside the spawned worker task. This avoids building copied
sample payload jobs in the main thread and keeps worker contexts reusable.

## Attempt 01

`attempt_01_metropolis_s25_beta2p5_lr0005_b64_chunk2_seed2468_e20`

- dynamics: Metropolis
- radius: `r=8`
- sweeps: `25/25`, reads: `3/3`
- beta: `2.5`
- learning rates: `5e-4` for W0/W12/W2O, `5e-5` for bias
- output bias training: false
- batch size: `64`
- job chunk size: `2`
- epochs: `20`
- train/test per class: `100/20`
- workers/threads: `32/32`

Watch for:

- non-flat prediction counts;
- test accuracy staying above chance for multiple epochs;
- train accuracy increasing without skipped samples exploding.

Result: best test accuracy `0.21` at epoch 1. The run did not sustain
learning; it collapsed into near-single-class predictions by epoch 10 and ended
at `0.08` test accuracy with prediction counts `195-5-0-0-0-0-0-0-0-0`.

## Attempt 02

`attempt_02_metropolis_s25_beta2p5_lr0001_b128_chunk4_seed2468_e40`

Same dynamics and nudge scale as attempt 01, but lower Adam learning rate and a
larger minibatch:

- dynamics: Metropolis
- radius: `r=8`
- sweeps: `25/25`, reads: `3/3`
- beta: `2.5`
- learning rates: `1e-4` for W0/W12/W2O, `1e-5` for bias
- output bias training: false
- batch size: `128`
- job chunk size: `4`
- epochs: `40`

Purpose: test whether attempt 01's class-attractor collapse is caused by update
size/noisy minibatches rather than by the r8 sampler itself.

Result: successful direction, but not stable enough. Best test accuracy reached
`0.415`; final test accuracy was `0.245`. Prediction counts stayed more diverse
than attempt 01 for many epochs, then drifted toward fewer classes. This supports
the optimizer-instability hypothesis.

## Attempt 03

`attempt_03_metropolis_s25_beta2p5_lr00005_b128_chunk4_seed2468_e80`

Same as attempt 02 but another `2x` lower learning rate and longer run:

- dynamics: Metropolis
- radius: `r=8`
- sweeps: `25/25`, reads: `3/3`
- beta: `2.5`
- learning rates: `5e-5` for W0/W12/W2O, `5e-6` for bias
- output bias training: false
- batch size: `128`
- job chunk size: `4`
- epochs: `80`

Purpose: test whether the attempt 02 peak/decay can be turned into slower,
sustained learning.

Result: stronger, but still unstable. Best test accuracy reached `0.535`
around the middle of the run; final test accuracy decayed to `0.21`. This shows
that `r=8` can learn from scratch under the current manager path, but the
optimizer/readout dynamics still drift into class-attractor states if training
continues past the useful window.

## Langevin Smoke Test

`diagnostics/smoke_local_langevin_s05`

Local Langevin is not currently a valid drop-in branch for this experiment:
`LocalLangevin` requires `Continuous` active layers, while the single-hidden
local MNIST graph uses `Discrete` spin layers. A Langevin comparison therefore
requires a separate continuous-state experiment variant, not just changing
`ISING_MNIST_PM_DYNAMICS`.

## Next Queue

The next suite keeps the self-loading manager worker path and tests only
changes that address the observed peak-then-decay behavior:

- lower Metropolis learning rate at the same `β=2.5`;
- lower nudge strength with a conservative learning rate;
- two additional random initializations of the best Metropolis recipe;
- a higher-relaxation variant to check whether late collapse is a sampling
  issue or mainly an optimizer/readout issue.

The queued runs live in `attempt_04` through `attempt_08`, and the launcher is
`_launchers/launch_next_r8_rescue_suite.ps1`.

## Retry Suite 02

The first rescue suite found that the most stable branch was
`β=1.5`, `lr=2.5e-5`, `batchsize=128`, `job_chunk_size=4`, `25/25` sweeps.
The next retry keeps the same from-scratch setup and tests whether the remaining
collapse is mainly from nudge strength or optimizer drift:

- `attempt_09`: repeat the best `β=1.5`, `lr=2.5e-5` recipe longer.
- `attempt_10`: lower nudge to `β=1.0` at the same learning rate.
- `attempt_11`: keep `β=1.5` but lower learning rate to `1e-5`.
- `attempt_12`: combine `β=1.0` with `lr=1e-5`.

Launcher: `_launchers/launch_r8_retry_suite_02.ps1`.

Important: the long attempt names in retry suite 02 hit Windows path limits when
checkpointing `best_params.bin`, so some runs ended early. Later suites use short
attempt directory names.

## Retry Suite 03

This suite tests the user's requested combination: more relaxation with smaller
nudging. All attempt folders are intentionally short to avoid Windows path
length issues during checkpoint saves.

- `a13_s35_b15_lr25_e160`: `35/35` sweeps, `β=1.5`, `lr=2.5e-5`.
- `a14_s35_b10_lr25_e160`: `35/35` sweeps, `β=1.0`, `lr=2.5e-5`.
- `a15_s50_b10_lr10_e200`: `50/50` sweeps, `β=1.0`, `lr=1e-5`.
- `a16_s50_b075_lr10_e200`: `50/50` sweeps, `β=0.75`, `lr=1e-5`.

Launcher: `_launchers/launch_r8_more_sweeps_small_beta.ps1`.
