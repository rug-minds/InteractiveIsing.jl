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

- epochs: `2400`
- free/nudged relaxation: `300` / `300`
- beta: `2.0`
- temperature: `0.005`
- stepsize: `0.4`
- nudged derivative samples: `1` every `1` full sweep(s)
- Minit/eval repeats: `16` / `16`
- workers: `8`

Best logged result:

- epoch: `1600`
- MSE: `0.13680858639804466`
- accuracy: `1.0`
- means: `[-0.5667815964842025, 0.6234632494498418, 0.6427677622880309, -0.6997311799872832]`

CSV: `timeavg_learning_metrics.csv`
Plot: `timeavg_learning_progress.png`
