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

- epochs: `1200`
- free/nudged relaxation: `600` / `600`
- beta: `2.0`
- temperature: `0.005`
- validation temperature: `0.005`
- stepsize: `0.4`
- nudged state-average samples: `1` every `1` full sweep(s)
- Minit/eval repeats: `8` / `16`
- workers: `8`

Best logged result:

- epoch: `1200`
- MSE: `0.5616152892553663`
- accuracy: `0.75`
- means: `[-0.5833443381700856, -0.005519295185077787, 0.6670922050323872, -0.024826905853710737]`

CSV: `timeavg_learning_metrics.csv`
Plot: `timeavg_learning_progress.png`
