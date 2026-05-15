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
- beta: `1.0`
- temperature: `0.1`
- validation temperature: `0.005`
- stepsize: `1.0`
- nudged state-average samples: `16` every `1` full sweep(s)
- Minit/eval repeats: `8` / `24`
- workers: `8`

Best logged result:

- epoch: `2000`
- MSE: `1.0747699048383894`
- accuracy: `0.75`
- means: `[-0.5366916006784509, 0.9193539092029633, 0.3827956835473887, 0.9227532307249109]`

CSV: `timeavg_learning_metrics.csv`
Plot: `timeavg_learning_progress.png`
