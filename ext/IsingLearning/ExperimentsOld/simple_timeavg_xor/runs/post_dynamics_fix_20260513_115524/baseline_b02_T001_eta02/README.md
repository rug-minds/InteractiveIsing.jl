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

- epochs: `300`
- free/nudged relaxation: `80` / `80`
- beta: `0.2`
- temperature: `0.01`
- stepsize: `0.2`
- Minit/eval repeats: `4` / `8`
- workers: `8`

Best logged result:

- epoch: `50`
- MSE: `1.0935682600930847`
- accuracy: `0.5`
- means: `[-0.42357361162951773, -0.35436296434512116, 0.5536922747204601, 0.4172212369029048]`

CSV: `timeavg_learning_metrics.csv`
Plot: `timeavg_learning_progress.png`
