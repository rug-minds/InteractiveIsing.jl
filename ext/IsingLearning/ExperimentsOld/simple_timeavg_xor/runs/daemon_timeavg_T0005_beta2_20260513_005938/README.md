# Simple 2->4->1 Time-Averaged XOR Learning

This run relearns the physical `2 -> 4 -> 1` XOR graph. It does not
reuse a learned graph. EqProp worker trajectories and validation
trajectories are launched and synchronized with `ProcessManager`.

Validation classifies by averaging the scalar output spin after a
burn-in period:

- burn-in full sweeps: `600`
- averaged samples: `50`
- sampling interval: one sample every `1` full sweep(s)

Training settings:

- epochs: `800`
- free/nudged relaxation: `600` / `600`
- beta: `2.0`
- temperature: `0.005`
- stepsize: `0.4`
- Minit/eval repeats: `8` / `16`
- workers: `8`

Best logged result:

- epoch: `800`
- MSE: `1.7608284843634476`
- accuracy: `0.5`
- means: `[-0.9201846093935023, -0.9314708856478408, 0.9368196728495412, 0.8172429405646425]`

CSV: `timeavg_learning_metrics.csv`
Plot: `timeavg_learning_progress.png`
