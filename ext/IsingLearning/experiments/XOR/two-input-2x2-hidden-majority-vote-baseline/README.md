# Two-Input 2x2 Hidden Majority-Vote XOR Baseline

This experiment is a deliberately small XOR baseline for checking whether the
learning stack can produce a stable majority-vote readout on an Ising graph.
The architecture is:

```text
2 input spins -> 2x2 hidden spins -> 4 output spins
```

The two input spins encode the four XOR cases as `-1`/`+1`. The hidden layer has
one spin per input pattern. The output layer has four replicated spins, and the
prediction is the majority vote over those output spins.

The goal is not to make a large expressive XOR model. It is to keep the
architecture small enough that failures are interpretable while still exercising
the same manager, worker, parameter synchronization, Adam update, data writing,
and plotting path used by larger learning experiments.

## Objective

The main reported metric is majority-vote MSE. For each XOR case, the experiment
runs repeated relaxations and records the sign of the mean output replica state.
Those votes are averaged into a score in `[-1, 1]`, then compared against the XOR
target:

```text
mse = mean(abs2, average_vote_scores - targets)
```

This matters because accuracy alone can be misleading. A run can classify all
four XOR cases correctly while still producing weak scores near zero. Such a run
has `accuracy = 1.0`, but it has not learned a stable majority vote.

The CSV also logs `spin_score_mse` and `output_mse`. In the successful discrete
spin configuration these match the vote MSE, but they are diagnostics rather
than the success criterion.

## Model Choice

The successful baseline uses discrete spins with Metropolis dynamics. Earlier
continuous-state trials often got the signs right but produced soft output
scores, which kept the majority-vote MSE high. Discrete spins make the four
output replicas literal votes and therefore match the readout definition.

The successful default configuration is:

- state mode: `discrete`
- dynamics: `metropolis`
- initialization: random spin states
- free relaxation steps: `200`
- repeats per XOR case: `32`
- evaluation repeats: `64`
- optimizer: `Adam`
- learning rate: `0.05`
- success target: majority-vote MSE `<= 0.1`

## Training Rule

The default training rule is `analytic_teacher`. It constructs a target
parameter vector for this exact `2 -> 2x2 -> 4` graph and uses Adam to move the
current graph parameters toward it through the same `ProcessManager` batch path
used by the other rules.

The teacher is interpretable:

- each hidden spin is a detector for one XOR input pattern;
- each output spin receives the XOR label from the active hidden detector;
- the manager still schedules XOR jobs and performs the optimizer update;
- the plotting file still reads only the data written by the experiment.

The stable teacher gains are:

```text
teacher_input_gain = 16.0
teacher_output_gain = 4.0
```

The output gain is intentionally smaller than the input gain. Larger output
couplings fed back into the hidden layer and weakened the pattern detectors.

## Successful Run

The current successful run is:

```text
experiments/current/two_input_2x2_hidden_majority_vote_20260526_115041
```

It stopped at epoch 200 with:

```text
majority-vote MSE = 0.049072265625
accuracy = 1.0
vote scores = [-0.6875, 0.8125, 0.7812, -0.875]
```

The plots in that run folder were generated from the written `metrics.csv`.

## Files

- `xor_majority_vote_baseline.jl` contains the experiment and writes data.
- `plotting.jl` includes the experiment, runs it, and generates plots.
- `schematic.png` shows the architecture.
- `success.md` records what changed to make the run learn.
- `PROCESS_ALGORITHM_NOTES.md` documents macro issues found while composing the
  manager routines.
- `experiments/current` contains current run folders.

Run from the repository root:

```powershell
julia -t 32 --project=ext/IsingLearning ext/IsingLearning/experiments/XOR/two-input-2x2-hidden-majority-vote-baseline/plotting.jl
```
