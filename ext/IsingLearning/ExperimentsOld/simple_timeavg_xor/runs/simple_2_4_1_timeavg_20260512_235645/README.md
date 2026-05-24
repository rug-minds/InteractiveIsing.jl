# Simple 2->4->1 Time-Averaged XOR Learning

This run relearns the physical `2 -> 4 -> 1` XOR graph. It does not
reuse a learned graph. EqProp worker trajectories and validation
trajectories are launched and synchronized with `ProcessManager`.

Validation classifies by averaging the scalar output spin after a
burn-in period:

- burn-in full sweeps: `1`
- averaged samples: `2`
- sampling interval: one sample every `1` full sweep(s)

Training settings:

- epochs: `2`
- free/nudged relaxation: `2` / `2`
- beta: `0.2`
- temperature: `0.01`
- stepsize: `0.2`
- Minit/eval repeats: `1` / `1`
- workers: `2`

Best logged result:

- epoch: `0`
- MSE: `0.8760548342853713`
- accuracy: `0.5`
- means: `[0.03462816711237035, 0.12503846035698474, -0.28768343051899464, -0.8996128500676477]`

CSV: `timeavg_learning_metrics.csv`
Plot: `timeavg_learning_progress.png`
