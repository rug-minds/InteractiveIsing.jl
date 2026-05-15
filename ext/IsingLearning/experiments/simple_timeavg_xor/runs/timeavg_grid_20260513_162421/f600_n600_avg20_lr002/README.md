# Simple 2->4->1 Time-Averaged XOR Learning

This run relearns the physical `2 -> 4 -> 1` XOR graph. It does not
reuse a learned graph. EqProp worker trajectories and validation
trajectories are launched and synchronized with `ProcessManager`.

Validation classifies by averaging the scalar output spin after a
burn-in period:

- burn-in full sweeps: `20`
- averaged samples: `80`
- sampling interval: one sample every `1` full sweep(s)

Training settings:

- epochs: `2`
- free/nudged relaxation: `600` / `600`
- beta: `2.0`
- temperature: `0.005`
- stepsize: `0.4`
- nudged derivative samples: `20` every `1` full sweep(s)
- Minit/eval repeats: `8` / `16`
- workers: `2`

Best logged result:

- epoch: `1`
- MSE: `1.2744524338883028`
- accuracy: `0.5`
- means: `[-0.8075889848159228, -0.68915035273928, 0.5660439564632083, 0.42099999117090203]`

CSV: `timeavg_learning_metrics.csv`
Plot: `timeavg_learning_progress.png`
