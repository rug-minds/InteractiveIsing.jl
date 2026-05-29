# Simple 2->4->1 Time-Averaged XOR Learning

This run relearns the physical `2 -> 4 -> 1` XOR graph. It does not
reuse a learned graph. EqProp worker trajectories and validation
trajectories are launched and synchronized with `ProcessManager`.

Validation classifies by averaging the scalar output spin after a
burn-in period:

- burn-in full sweeps: `8`
- averaged samples: `50`
- sampling interval: one sample every `1` full sweep(s)

Training settings:

- epochs: `800`
- free/nudged relaxation: `80` / `80`
- beta: `0.2`
- temperature: `0.01`
- stepsize: `0.2`
- Minit/eval repeats: `4` / `8`
- workers: `8`

Best logged result:

- epoch: `500`
- MSE: `1.0556307726928726`
- accuracy: `0.5`
- means: `[-0.6530288074656677, -0.3204677342739624, 0.3704254743852856, 0.4007622788186402]`

CSV: `timeavg_learning_metrics.csv`
Plot: `timeavg_learning_progress.png`
