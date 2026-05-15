# Simple 2->4->1 Time-Averaged XOR Learning

This run relearns the physical `2 -> 4 -> 1` XOR graph. It does not
reuse a learned graph. EqProp worker trajectories and validation
trajectories are launched and synchronized with `ProcessManager`.

Validation classifies by averaging the scalar output spin after a
burn-in period:

- burn-in full sweeps: `40`
- averaged samples: `80`
- sampling interval: one sample every `1` full sweep(s)

Training settings:

- epochs: `1200`
- free/nudged relaxation: `1200` / `1200`
- beta: `4.0`
- temperature: `0.002`
- stepsize: `0.8`
- nudged derivative samples: `30` every `1` full sweep(s)
- Minit/eval repeats: `8` / `16`
- workers: `8`

Best logged result:

- epoch: `1100`
- MSE: `1.1453530677404442`
- accuracy: `0.75`
- means: `[-0.493853708677403, 0.04124898179179424, -0.8391716739068423, -0.8467934670923235]`

CSV: `timeavg_learning_metrics.csv`
Plot: `timeavg_learning_progress.png`
