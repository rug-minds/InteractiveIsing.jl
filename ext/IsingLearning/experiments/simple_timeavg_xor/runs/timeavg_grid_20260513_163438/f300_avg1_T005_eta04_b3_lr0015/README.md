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

- epochs: `1800`
- free/nudged relaxation: `300` / `300`
- beta: `3.0`
- temperature: `0.005`
- stepsize: `0.4`
- nudged derivative samples: `1` every `1` full sweep(s)
- Minit/eval repeats: `8` / `16`
- workers: `8`

Best logged result:

- epoch: `1400`
- MSE: `0.868296409901554`
- accuracy: `0.75`
- means: `[-0.1852250104776353, 0.09839473732198847, -0.00039200843912786526, -0.002176801323256379]`

CSV: `timeavg_learning_metrics.csv`
Plot: `timeavg_learning_progress.png`
