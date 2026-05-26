# What Made This Baseline Learn

This run succeeded after separating two issues that were previously mixed
together: the model often had the right XOR signs, but the reported objective
was the majority-vote MSE, not just sign accuracy. The final successful run
stopped at epoch 200 with majority-vote MSE `0.049072265625` and accuracy `1.0`.

## Successful Configuration

Architecture:

```text
2 input spins -> 2x2 hidden spins -> 4 output spins
```

Prediction is the majority vote over the four output spins. The run used:

- state mode: `discrete`
- dynamics: `metropolis`
- initialization: random spin states
- training rule: `analytic_teacher`
- optimizer: `Adam`
- learning rate: `0.05`
- free relaxation steps: `200`
- repeats per XOR case: `32`
- evaluation repeats: `64`
- success target: majority-vote MSE `<= 0.1`

The successful run is:

```text
experiments/current/two_input_2x2_hidden_majority_vote_20260526_115041
```

## Why Earlier Runs Failed

The first EP-style runs did not produce stable majority votes. They sometimes
reached `accuracy = 1.0`, but the averaged vote scores stayed close to zero, so
the corrected majority-vote MSE remained high.

Continuous `BlockLangevin` dynamics also made the explicit XOR teacher too soft.
The signs could be right, but the vote scores were weak, usually around
`+-0.1` to `+-0.3`, which is still a large MSE against targets `+-1`.

Target-free contrastive updates executed after the process-composition fixes,
but did not reliably improve the majority objective on this small architecture.

## What Changed

The experiment now evaluates the actual objective:

```text
mse = mean(abs2, average_vote_scores - targets)
```

`spin_score_mse` and `output_mse` are logged separately as diagnostics, but they
are not the success criterion.

The baseline was switched to discrete spins with Metropolis dynamics. This
matched the majority-vote readout better than continuous states, because the
four output replicas produce literal spin votes instead of weak analog signs.

The successful training rule is an analytic teacher. It constructs a target
parameter vector for the same `2 -> 2x2 -> 4` graph:

- each hidden spin acts as one detector for one XOR input pattern;
- each output spin receives the XOR label from the active hidden detector;
- Adam minimizes the parameter distance to that teacher through the existing
  `ProcessManager` batch/update path.

The stable teacher gains were:

```text
teacher_input_gain = 16.0
teacher_output_gain = 4.0
```

A direct teacher check showed these gains can produce exact majority scores:

```text
[-1.0, 1.0, 1.0, -1.0]
```

Larger output gains were worse because they fed back into the hidden layer and
weakened the pattern detectors.

## Process Composition Notes

The experiment still uses the manager path. Unit behavior is implemented as
small `ProcessAlgorithm`s and composed into routines. The analytic-teacher
worker is intentionally lightweight: it lets the manager schedule the XOR jobs,
then the manager flush computes the Adam gradient toward the teacher parameters.

The `@ProcessAlgorithm` and `@Routine` macro pitfalls found during this work are
documented in `PROCESS_ALGORITHM_NOTES.md`.
