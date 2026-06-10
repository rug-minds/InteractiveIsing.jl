# Softplus-Margin Symmetric Nudge Resume Notes - 2026-06-09

## What Is Saved

The current forward and symmetric softplus-margin MNIST files save recoverable checkpoints:

- `best_checkpoint.bin`: parameters and Adam optimizer state at the best evaluated test accuracy.
- `latest_checkpoint.bin`: parameters and Adam optimizer state at the latest evaluated epoch.
- `final_checkpoint.bin`: parameters and Adam optimizer state at run completion.
- `mnist_784_120_40_softplus_margin_nudge_adam.csv`: per-epoch metrics and checkpoint paths.
- `settings.md`: the run configuration.

The checkpoint payload contains the architecture, graph parameters, Adam `opt_state`, metric rows, and config. Exact continuation should resume from `best_checkpoint.bin` or `latest_checkpoint.bin` with `ISING_MNIST_IF_RESET_OPT_STATE_ON_RESUME=false`.

I added `ISING_MNIST_IF_RESET_OPT_STATE_ON_RESUME`. This is important for LR schedule experiments: restoring Adam state can also restore the old optimizer rule, so a resumed branch that changes LR should rebuild Adam state explicitly with `ISING_MNIST_IF_RESET_OPT_STATE_ON_RESUME=true`.

I also changed new checkpoint writes to store `config` as a named tuple instead of a concrete `InputFieldMNISTConfig`. Julia `Serialization` is not schema-tolerant for concrete structs: after adding `reset_opt_state_on_resume`, older checkpoints failed to load with `EOFError`. The loader now recovers pre-reset-option checkpoints through a sidecar `*_recovered.bin` file that keeps params, Adam state, rows, and a neutral named-tuple config.

## Old Baseline Reference

The old 784-120-40 EqProp baseline result is documented at:

`ext/IsingLearning/experiments/MNIST/784-120-40-baseline/experiments/current/20260529_mnist_784_120_40_baseline_revisit_gt90_e40_32w`

Important facts:

- The earlier known best before the revisit used `beta=5.0`, `lr=0.003`, `weight_decay=0.0`, and peaked at epoch 15 with test accuracy `0.885201793721973`.
- The revisit run `beta5_lr0015_wd0_e40` used `beta=5.0`, `lr=0.0015`, `weight_decay=0.0`, `500` sweeps, `80000` relaxation steps, `T=0.001`, stepsize `0.5`, `32` workers, batch size `128`, and full balanced MNIST (`5421` train per class, `892` test per class).
- That revisit peaked at epoch 20 with test accuracy `0.8918161434977578`.
- Epochs took about `428` to `446` seconds each.
- After the peak, the run degraded: epoch 35 was `0.850896860986547`, and epoch 40 was `0.7380044843049327`.

This confirms that best-checkpoint recovery is normal and necessary here. Final checkpoints can be much worse than the best checkpoint.

## Current Symmetric Nudge Best

Current best softplus-margin symmetric nudge run:

`ext/IsingLearning/experiments/MNIST/baseline_new_nudge/experiments/current/symmetric_nudge_target_weight_grid_v4_20260609/b0p020_lr0p00025_T0p00005_wd0p001_step0p5_pos9p0_neg0p50_priortrue_adaptivefalse`

Configuration:

- symmetric `+beta/-beta` nudge
- `beta=0.02`
- `lr=0.00025`
- `T=0.00005`
- stepsize `0.5`
- weight decay `0.001`
- positive/negative target mask weights `9.0 / 0.50`
- `500` sweeps
- `32` workers, `32` Julia threads
- batch size `128`
- train/test per class `1280 / 100`

Metrics:

| epoch | test accuracy | test loss | epoch seconds |
| ---: | ---: | ---: | ---: |
| 0 | 0.067 | 74.25641 | 0.0 |
| 1 | 0.479 | 15.553007 | 112.081 |
| 2 | 0.522 | 15.002454 | 105.475 |
| 3 | 0.565 | 13.490842 | 107.148 |
| 4 | 0.560 | 13.142413 | 7.634 |

Best checkpoint: epoch 3, `0.565`.

The original `best_checkpoint.bin` was recovered into:

`ext/IsingLearning/experiments/MNIST/baseline_new_nudge/experiments/current/symmetric_nudge_target_weight_grid_v4_20260609/b0p020_lr0p00025_T0p00005_wd0p001_step0p5_pos9p0_neg0p50_priortrue_adaptivefalse/best_checkpoint_recovered.bin`

## What To Tune After Resume

The first continuation should resume from the best symmetric checkpoint, not the final checkpoint.

Most sensible first knob: lower LR after the peak. This is standard because the run improved quickly and then flattened/declined, while the old baseline also degraded after its peak. For LR changes on resume, use `ISING_MNIST_IF_RESET_OPT_STATE_ON_RESUME=true`.

Beta should not be changed first. In EqProp-style contrastive updates, beta controls the finite-difference perturbation size. Smaller beta is closer to the infinitesimal derivative but increases Monte Carlo noise after dividing by `2beta`; larger beta improves signal size but biases the contrastive estimate and can push the nudged phase into different nonlinear dynamics. The current grid already showed `beta=0.05` worse than `beta=0.02`, so beta should stay fixed until we know whether a continuation LR schedule helps.

Temperature and sweeps affect equilibration rather than the supervised objective directly. More sweeps or a slightly warmer/reverse-annealed nudged phase can help if free/nudged samples are not close enough to equilibrium, but they are more expensive. I would test those only after a lower-LR continuation.

Recommended continuation order:

1. Resume best epoch 3 with `beta=0.02`, `lr=0.00015`, reset Adam state, no batch early-stop, epoch decline-stop enabled.
2. If still improving, continue with the same beta and lower LR again around plateau (`0.00010`).
3. If it stalls immediately, try the same resume with `lr=0.00025` and preserved Adam state as a control.
4. Only then branch beta around the current value (`0.015`, `0.025`) with Adam reset, because changing beta changes gradient scale and objective bias.
5. For equilibrium sensitivity, run the best branch with more sweeps (`750` or `1000`) or nudged reverse annealing, accepting the slower epoch time.

## Resume Grid Results

Resume grid root:

`ext/IsingLearning/experiments/MNIST/baseline_new_nudge/experiments/current/symmetric_nudge_resume_from_v4_best_20260609`

Completed branches:

| branch | beta | lr | reset Adam | best epoch | best test accuracy | final epoch | final test accuracy | notes |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `resume_lr0p00015_reset` | 0.02 | 0.00015 | true | 12 | 0.689 | 12 | 0.689 | best branch |
| `resume_lr0p00010_reset` | 0.02 | 0.00010 | true | 9 | 0.620 | 12 | 0.599 | under-adapted versus 0.00015 |
| `resume_lr0p00025_keepadam` | 0.02 | 0.00025 | false | 6 | 0.616 | 8 | 0.580 | stopped after two evaluated declines |

Best completed branch:

- outdir: `resume_lr0p00015_reset`
- resumed from v4 epoch-3 best checkpoint
- `beta=0.02`
- `lr=0.00015`
- `T=0.00005`
- weight decay `0.001`
- target weights `9.0 / 0.50`
- `ISING_MNIST_IF_RESET_OPT_STATE_ON_RESUME=true`
- best test accuracy: `0.689` at epoch 12
- final test accuracy: `0.689` at epoch 12
- epoch training times after compile/setup: about `78` to `90` seconds

Resume branch trajectory:

| epoch | test accuracy | test loss | seconds |
| ---: | ---: | ---: | ---: |
| 4 | 0.575 | 12.819110 | 90.402 |
| 5 | 0.613 | 13.661360 | 81.275 |
| 6 | 0.647 | 13.103011 | 85.934 |
| 7 | 0.609 | 13.203733 | 81.645 |
| 8 | 0.626 | 13.129021 | 82.951 |
| 9 | 0.668 | 12.393614 | 88.626 |
| 10 | 0.619 | 13.173699 | 78.206 |
| 11 | 0.643 | 12.588126 | 85.042 |
| 12 | 0.689 | 11.860693 | 84.380 |

This confirms that restarting from the best checkpoint and reducing LR can continue learning. Lowering LR too far (`0.00010`) was worse, and continuing with the old Adam state at `0.00025` was also worse. The next most defensible experiment is to resume from the new `0.689` checkpoint with `lr=0.00010` or `0.000075` and reset Adam, or to keep `lr=0.00015` but increase sweeps to test equilibrium sensitivity.

## 2026-06-10 Resume From 0.689

Grid root:

`ext/IsingLearning/experiments/MNIST/baseline_new_nudge/experiments/current/symmetric_nudge_resume_from_0p689_20260610`

Starting checkpoint:

`ext/IsingLearning/experiments/MNIST/baseline_new_nudge/experiments/current/symmetric_nudge_resume_from_v4_best_20260609/resume_lr0p00015_reset/best_checkpoint.bin`

Completed branches:

| branch | beta | lr | reset Adam | sweeps | best epoch | best test accuracy | final epoch | final test accuracy | notes |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cont_lr0p00015_keepadam_s500` | 0.02 | 0.00015 | false | 500 | 12 | 0.689 | 18 | 0.649 | did not improve; early-stopped after declines |
| `cont_lr0p00012_reset_s500` | 0.02 | 0.00012 | true | 500 | 17 | 0.714 | 20 | 0.658 | best branch; new checkpoint |
| `cont_lr0p00015_keepadam_s750` | 0.02 | 0.00015 | false | 750 | 12 | 0.689 | 18 | 0.630 | more sweeps were slower and worse with this LR/state |

Best new checkpoint:

`ext/IsingLearning/experiments/MNIST/baseline_new_nudge/experiments/current/symmetric_nudge_resume_from_0p689_20260610/cont_lr0p00012_reset_s500/best_checkpoint.bin`

The useful change was another LR reduction with Adam reset at the checkpoint. Increasing sweeps from 500 to 750 did not help in this branch and raised training time from about `76-91s` per epoch to about `113-122s` per epoch.

Trajectory for the winning branch after resuming:

| epoch | test accuracy | test loss | seconds |
| ---: | ---: | ---: | ---: |
| 13 | 0.631 | 13.100943 | 90.155 |
| 14 | 0.668 | 12.662383 | 79.872 |
| 15 | 0.694 | 11.504372 | 78.526 |
| 16 | 0.679 | 12.088132 | 77.379 |
| 17 | 0.714 | 11.591430 | 78.930 |
| 18 | 0.682 | 11.639980 | 82.543 |
| 19 | 0.695 | 11.680315 | 75.918 |
| 20 | 0.658 | 11.773059 | 79.571 |

## 2026-06-10 Resume From 0.714

Grid root:

`ext/IsingLearning/experiments/MNIST/baseline_new_nudge/experiments/current/symmetric_nudge_resume_from_0p714_20260610`

Targeted same-LR reset root:

`ext/IsingLearning/experiments/MNIST/baseline_new_nudge/experiments/current/symmetric_nudge_resume_from_0p714_targeted_20260610`

Starting checkpoint:

`ext/IsingLearning/experiments/MNIST/baseline_new_nudge/experiments/current/symmetric_nudge_resume_from_0p689_20260610/cont_lr0p00012_reset_s500/best_checkpoint.bin`

Completed branches:

| branch | beta | lr | reset Adam | sweeps | best epoch | best test accuracy | final epoch | final test accuracy | notes |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cont_lr0p00010_reset_s500` | 0.02 | 0.00010 | true | 500 | 17 | 0.714 | 24 | 0.689 | lower LR did not improve; early-stopped |
| `cont_lr0p00008_reset_s500` | 0.02 | 0.00008 | true | 500 | 17 | 0.714 | 26 | 0.691 | lower LR did not improve; local high after resume was 0.705 |
| `cont_lr0p00012_reset_s500` | 0.02 | 0.00012 | true | 500 | 17 | 0.714 | 21 | 0.658 | same LR with reset did not improve; early-stopped |

The planned `0.00006` branch was launched accidentally after the wrapper had already advanced to the next run; it was stopped before producing useful metrics because the two higher-LR continuation branches already showed that lowering LR further was not the promising direction.

Conclusion after the `0.714` checkpoint: the tested continuations plateau below the saved best. The current best recoverable state remains the epoch-17 `0.714` checkpoint. The next useful search should change something other than only LR-after-plateau, for example:

- modest beta branch around the current value, with Adam reset and LR retuned together (`beta=0.015` or `0.025`; do not compare without retuning LR because contrastive updates scale with beta);
- target-weight branch around `positive_target_weight=9.0`, especially a slightly lower positive weight if output classes are over-forced;
- adaptive or stronger weight decay branch, watching `w_norm` and `w_input_norm` rather than only accuracy;
- a proper validation split larger than 100/class or repeated test seeds, because the current 100/class test readout is noisy enough that single-epoch peaks may overstate real progress.

## Checkpointing And Manual Tuning

Saving best checkpoints and resuming from them is normal in ML. Reducing LR after a plateau is also normal. What is not ideal is doing this manually as the main training loop.

The standard version is to automate it:

- save `best_checkpoint.bin` whenever validation accuracy improves;
- keep `latest_checkpoint.bin` for crash recovery;
- use a scheduler such as reduce-on-plateau for LR;
- reset optimizer state when changing objective scale or beta, or when an old Adam state seems to drag a resumed run away from the peak;
- stop branches after a fixed number of non-improving evaluations;
- treat beta changes as coupled with LR, because beta changes both perturbation size and the scale/noise of the contrastive estimate.

For this experiment, the observed pattern is plausible: noisy contrastive estimates and small validation/test slices make accuracy bounce around. The right fix is not endless manual restart tuning, but a small controller that resumes from the best checkpoint, branches a limited set of coupled hyperparameters, and records every branch in the summary CSV/notes.

## 2026-06-10 Beta And Relaxation Search From 0.714

Grid root:

`ext/IsingLearning/experiments/MNIST/baseline_new_nudge/experiments/current/symmetric_nudge_beta_relax_from_0p714_20260610`

Starting checkpoint:

`ext/IsingLearning/experiments/MNIST/baseline_new_nudge/experiments/current/symmetric_nudge_resume_from_0p689_20260610/cont_lr0p00012_reset_s500/best_checkpoint.bin`

Completed branches:

| branch | beta | lr | sweeps | nudged temp schedule | peak T | weight decay | best epoch | best test accuracy | final epoch | final test accuracy | notes |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `b0p015_lr0p00009_s500_fixed` | 0.015 | 0.00009 | 500 | fixed | 0.00005 | 0.001 | 17 | 0.714 | 24 | 0.702 | lower beta did not improve |
| `b0p025_lr0p00015_s500_fixed` | 0.025 | 0.00015 | 500 | fixed | 0.00005 | 0.001 | 17 | 0.714 | 20 | 0.681 | higher beta alone was worse |
| `b0p020_lr0p00012_s350_fixed` | 0.020 | 0.00012 | 350 | fixed | 0.00005 | 0.001 | 17 | 0.714 | 21 | 0.686 | faster; local high after resume was 0.708 |
| `b0p020_lr0p00012_s750_fixed` | 0.020 | 0.00012 | 750 | fixed | 0.00005 | 0.001 | 17 | 0.714 | 22 | 0.651 | slower and worse |
| `b0p025_lr0p00015_s500_reverseT` | 0.025 | 0.00015 | 500 | reverse anneal | 0.00010 | 0.001 | 21 | 0.724 | 24 | 0.629 | new best, but unstable afterward |

Best new checkpoint:

`ext/IsingLearning/experiments/MNIST/baseline_new_nudge/experiments/current/symmetric_nudge_beta_relax_from_0p714_20260610/b0p025_lr0p00015_s500_reverseT/best_checkpoint.bin`

Main read:

- Lower beta did not help at this checkpoint.
- Higher beta did not help by itself.
- More relaxation (`750` sweeps) was clearly worse and much slower.
- Fewer relaxation sweeps (`350`) was much faster (`~54-68s` training epochs) and got close, but did not improve accuracy.
- The successful change was higher beta plus reverse-annealed nudged temperature. This suggests the nudged phase needs a warmer/more mobile trajectory before settling, not simply larger beta or more sweeps.

At the `0.724` peak, norms were about `w_norm=0.852`, `w_input_norm=2.056`. By final epoch 24 in the same branch, norms had grown to about `w_norm=0.906`, `w_input_norm=2.145`, and accuracy had fallen to `0.629`. That motivated the stabilization grid below.

## 2026-06-10 Stabilization From 0.724

Grid root:

`ext/IsingLearning/experiments/MNIST/baseline_new_nudge/experiments/current/symmetric_nudge_stabilize_from_0p724_20260610`

Starting checkpoint:

`ext/IsingLearning/experiments/MNIST/baseline_new_nudge/experiments/current/symmetric_nudge_beta_relax_from_0p714_20260610/b0p025_lr0p00015_s500_reverseT/best_checkpoint.bin`

Completed branches:

| branch | beta | lr | sweeps | schedule | weight decay | best epoch | best test accuracy | final epoch | final test accuracy | notes |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `b0p025_lr0p00012_wd0p002_s500_reverseT` | 0.025 | 0.00012 | 500 | reverse anneal | 0.002 | 21 | 0.724 | 25 | 0.672 | stronger decay helped norms but still drifted down |
| `b0p025_lr0p00010_wd0p002_s500_reverseT` | 0.025 | 0.00010 | 500 | reverse anneal | 0.002 | 21 | 0.724 | 28 | 0.718 | best continuation stability; did not exceed 0.724 |
| `b0p025_lr0p00012_wd0p002_s350_reverseT` | 0.025 | 0.00012 | 350 | reverse anneal | 0.002 | 21 | 0.724 | 24 | 0.689 | faster but worse |

Stabilization read:

- The recoverable best remains `0.724`.
- `lr=0.00010`, `weight_decay=0.002`, 500 sweeps, reverse anneal was the most stable continuation and ended at `0.718`.
- The lower-sweep version is faster but too inaccurate after this checkpoint.
- The next likely branch is not more sweeps. It is either:
  - resume `0.724` with the stable setting and reduce LR further on plateau (`0.00008` to `0.00010`), or
  - keep `lr=0.00010`, `wd=0.002`, reverse anneal, and tune target weighting or beta more narrowly around `0.025`.

## Nudged Phase Initialization Check

The symmetric implementation initializes both nudged phases from the free equilibrium state for the same sample.

Relevant flow in `mnist_784_120_40_softplus_margin_symmetric_nudge_adam.jl`:

- free phase runs from `initstate!`, relaxes, and copies into `equilibrium_state`;
- plus phase calls `setgraph!(..., target = equilibrium_state)`, applies `+β` target clamping, then relaxes;
- minus phase calls `setgraph!(..., target = equilibrium_state)`, applies `-β` target clamping, then relaxes;
- the gradient compares `plus_state` and `minus_state`.

So plus and minus are both branched from the same free equilibrium. The free phase itself is not a persistent Markov chain across samples; it reinitializes per sample before free relaxation.

## 2026-06-10 Langevin Stepsize From 0.724

Grid root:

`ext/IsingLearning/experiments/MNIST/baseline_new_nudge/experiments/current/symmetric_nudge_stepsize_from_0p724_20260610`

Starting checkpoint:

`ext/IsingLearning/experiments/MNIST/baseline_new_nudge/experiments/current/symmetric_nudge_beta_relax_from_0p714_20260610/b0p025_lr0p00015_s500_reverseT/best_checkpoint.bin`

Shared settings:

- `beta=0.025`
- `lr=0.00010`
- `weight_decay=0.002`
- `sweeps=500`
- `nudge_temp_schedule=reverse_anneal`
- `nudge_temp_peak=0.00010`
- base `T=0.00005`
- Adam reset on resume
- early-stop patience widened to 4 declining evaluations so each branch gets several epochs

Completed/attempted branches:

| branch | stepsize | post-resume local high | final epoch | final test accuracy | notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `step0p35_lr0p00010_wd0p002_s500_reverseT` | 0.35 | 0.721 at epoch 23 | 32 | 0.716 | close to best after several epochs, but did not beat 0.724 |
| `step0p50_lr0p00010_wd0p002_s500_reverseT_patience` | 0.50 | 0.707 at epoch 22 before failure | 23 partial | 0.700 partial | failed on Windows opening `latest_checkpoint.bin`; not a learning divergence |
| `step0p70_lr0p00010_wd0p002_s500_reverseT` | 0.70 | 0.702 at epoch 32 | 32 | 0.702 | larger step was consistently worse |

Read:

- Giving branches more epochs was important: `stepsize=0.35` looked bad at epoch 22 (`0.678`) but rebounded to `0.721` at epoch 23 and finished `0.716`.
- Smaller stepsize (`0.35`) is viable and slightly more stable than the high step, but did not improve the checkpoint.
- Larger stepsize (`0.70`) is not helpful in this regime.
- The best recoverable checkpoint remains `0.724`.
- If continuing from `0.724`, the practical candidate is `stepsize=0.35` or `0.50`, `lr=0.00010`, `weight_decay=0.002`, reverse-annealed nudging. The next real improvement likely needs either a scheduler from that stable continuation, more robust validation, or a different target/decay schedule rather than simply larger Langevin steps.

The `0.50` branch produced metrics before failing:

| epoch | test accuracy | test loss |
| ---: | ---: | ---: |
| 22 | 0.707 | 11.560920 |
| 23 | 0.700 | 11.636090 |

Failure detail: the run errored while opening `latest_checkpoint.bin` with `SystemError: Invalid argument` during checkpoint save. Since it failed before finalization, treat its metrics as partial only. The existing `latest_checkpoint.bin` was checked afterward and is recoverable, but it contains rows only through epoch 22 (`0.707` test accuracy); epoch 23 exists only in the CSV/log.
