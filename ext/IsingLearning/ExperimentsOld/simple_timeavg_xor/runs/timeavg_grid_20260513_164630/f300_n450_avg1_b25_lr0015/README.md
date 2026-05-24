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
- free/nudged relaxation: `300` / `450`
- beta: `2.5`
- temperature: `0.005`
- validation temperature: `0.005`
- stepsize: `0.4`
- nudged derivative samples: `1` every `1` full sweep(s)
- Minit/eval repeats: `8` / `16`
- workers: `8`

Best logged result:

- epoch: `1600`
- MSE: `0.9081159933207873`
- accuracy: `1.0`
- means: `[-0.09874521000577767, 0.004265810266836721, 0.002673093997033138, -0.08673320286205027]`

CSV: `timeavg_learning_metrics.csv`
Plot: `timeavg_learning_progress.png`
