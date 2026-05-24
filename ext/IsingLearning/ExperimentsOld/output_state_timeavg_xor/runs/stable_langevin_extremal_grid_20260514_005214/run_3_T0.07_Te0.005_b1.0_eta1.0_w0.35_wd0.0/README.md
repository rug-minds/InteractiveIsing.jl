# Simple 2->4->1 Time-Averaged XOR Learning

This run relearns the physical `2 -> 4 -> 1` XOR graph. It does not
reuse a learned graph. EqProp worker trajectories and validation
trajectories are launched and synchronized with `ProcessManager`.

Validation classifies by averaging the scalar output spin after a
burn-in period:

- burn-in full sweeps: `40`
- averaged samples: `100`
- sampling interval: one sample every `1` full sweep(s)

Training settings:

- epochs: `2500`
- free/nudged relaxation: `260` / `260`
- beta: `1.0`
- temperature: `0.07`
- validation temperature: `0.005`
- stepsize: `1.0`
- nudged state-average samples: `16` every `1` full sweep(s)
- Minit/eval repeats: `8` / `24`
- workers: `8`

Best logged result:

- epoch: `1750`
- MSE: `1.049211296494373`
- accuracy: `0.75`
- means: `[-0.536007707373339, 0.9225682840910158, 0.458433764560846, 0.9189233127887628]`

CSV: `timeavg_learning_metrics.csv`
Plot: `timeavg_learning_progress.png`
