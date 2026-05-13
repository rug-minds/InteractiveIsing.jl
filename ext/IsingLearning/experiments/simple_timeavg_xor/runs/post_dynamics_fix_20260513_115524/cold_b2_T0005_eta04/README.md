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
- beta: `2.0`
- temperature: `0.005`
- stepsize: `0.4`
- Minit/eval repeats: `4` / `8`
- workers: `8`

Best logged result:

- epoch: `250`
- MSE: `1.2411116607299757`
- accuracy: `0.5`
- means: `[-0.8860731835117486, -0.7292797409848736, 0.5046245936112156, 0.30983285482125156]`

CSV: `timeavg_learning_metrics.csv`
Plot: `timeavg_learning_progress.png`
