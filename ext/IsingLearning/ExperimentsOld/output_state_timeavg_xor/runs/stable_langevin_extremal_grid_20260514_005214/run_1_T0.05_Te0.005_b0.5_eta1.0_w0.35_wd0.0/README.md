# Simple 2->4->1 Time-Averaged XOR Learning

This run relearns the physical `2 -> 4 -> 1` XOR graph. It does not
reuse a learned graph. EqProp worker trajectories and validation
trajectories are launched and synchronized with `ProcessManager`.

Validation classifies by averaging the scalar output spin after a
burn-in period:

- burn-in full sweeps: `40`
- averaged samples: `100`
- sampling interval: one sample every `1` full sweep(s)

Training settings:

- epochs: `2500`
- free/nudged relaxation: `260` / `260`
- beta: `0.5`
- temperature: `0.05`
- validation temperature: `0.005`
- stepsize: `1.0`
- nudged state-average samples: `16` every `1` full sweep(s)
- Minit/eval repeats: `8` / `24`
- workers: `8`

Best logged result:

- epoch: `500`
- MSE: `1.0552803692166428`
- accuracy: `0.75`
- means: `[-0.516221935843962, 0.9239120195952094, 0.4608967122144958, 0.9211086710138985]`

CSV: `timeavg_learning_metrics.csv`
Plot: `timeavg_learning_progress.png`
