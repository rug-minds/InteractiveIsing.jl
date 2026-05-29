# Simple 2->4->1 Time-Averaged XOR Learning

This run relearns the physical `2 -> 4 -> 1` XOR graph. It does not
reuse a learned graph. EqProp worker trajectories and validation
trajectories are launched and synchronized with `ProcessManager`.

Validation classifies by averaging the scalar output spin after a
burn-in period:

- burn-in full sweeps: `2`
- averaged samples: `4`
- sampling interval: one sample every `1` full sweep(s)

Training settings:

- epochs: `2`
- free/nudged relaxation: `20` / `20`
- beta: `0.2`
- temperature: `0.01`
- stepsize: `0.2`
- Minit/eval repeats: `1` / `1`
- workers: `2`

Best logged result:

- epoch: `0`
- MSE: `0.72605476009068`
- accuracy: `0.75`
- means: `[-0.07924428503044839, 0.04687782179667749, -0.06936258784582992, -0.9332937570668717]`

CSV: `timeavg_learning_metrics.csv`
Plot: `timeavg_learning_progress.png`
