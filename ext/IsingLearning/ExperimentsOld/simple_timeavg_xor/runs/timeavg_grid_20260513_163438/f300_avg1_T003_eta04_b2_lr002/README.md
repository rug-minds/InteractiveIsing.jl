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
- free/nudged relaxation: `300` / `300`
- beta: `2.0`
- temperature: `0.003`
- stepsize: `0.4`
- nudged derivative samples: `1` every `1` full sweep(s)
- Minit/eval repeats: `8` / `16`
- workers: `8`

Best logged result:

- epoch: `1800`
- MSE: `0.18170149362763047`
- accuracy: `1.0`
- means: `[-0.5989724470602561, 0.4858294588706564, 0.650028960307952, -0.5767603488503141]`

CSV: `timeavg_learning_metrics.csv`
Plot: `timeavg_learning_progress.png`
