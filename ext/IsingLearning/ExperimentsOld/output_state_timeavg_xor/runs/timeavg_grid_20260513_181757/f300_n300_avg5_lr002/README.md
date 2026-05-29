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
- free/nudged relaxation: `300` / `300`
- beta: `2.0`
- temperature: `0.005`
- validation temperature: `0.005`
- stepsize: `0.4`
- nudged state-average samples: `5` every `1` full sweep(s)
- Minit/eval repeats: `8` / `16`
- workers: `8`

Best logged result:

- epoch: `1200`
- MSE: `0.3965400780114502`
- accuracy: `0.75`
- means: `[-0.5760620775577322, -0.07738594049680839, 0.6643924727840749, -0.635247929605665]`

CSV: `timeavg_learning_metrics.csv`
Plot: `timeavg_learning_progress.png`
