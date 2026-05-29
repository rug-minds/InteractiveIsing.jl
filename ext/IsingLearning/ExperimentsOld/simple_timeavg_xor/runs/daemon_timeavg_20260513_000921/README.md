# Simple 2->4->1 Time-Averaged XOR Learning

This run relearns the physical `2 -> 4 -> 1` XOR graph. It does not
reuse a learned graph. EqProp worker trajectories and validation
trajectories are launched and synchronized with `ProcessManager`.

Validation classifies by averaging the scalar output spin after a
burn-in period:

- burn-in full sweeps: `300`
- averaged samples: `50`
- sampling interval: one sample every `1` full sweep(s)

Training settings:

- epochs: `400`
- free/nudged relaxation: `300` / `300`
- beta: `0.2`
- temperature: `0.05`
- stepsize: `0.4`
- Minit/eval repeats: `8` / `16`
- workers: `8`

Best logged result:

- epoch: `150`
- MSE: `1.2157443199012317`
- accuracy: `0.5`
- means: `[-0.7319835132829557, -0.3203927075829641, 0.5793043610471359, 0.6943207251227701]`

CSV: `timeavg_learning_metrics.csv`
Plot: `timeavg_learning_progress.png`
