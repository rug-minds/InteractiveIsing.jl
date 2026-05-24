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

- epochs: `2`
- free/nudged relaxation: `600` / `600`
- beta: `2.0`
- temperature: `0.005`
- stepsize: `0.4`
- nudged derivative samples: `1` every `1` full sweep(s)
- Minit/eval repeats: `8` / `16`
- workers: `2`

Best logged result:

- epoch: `2`
- MSE: `1.4439198434297271`
- accuracy: `0.5`
- means: `[-0.7960661086773932, -0.7573388914883898, 0.4801309308531656, 0.541293778971319]`

CSV: `timeavg_learning_metrics.csv`
Plot: `timeavg_learning_progress.png`
