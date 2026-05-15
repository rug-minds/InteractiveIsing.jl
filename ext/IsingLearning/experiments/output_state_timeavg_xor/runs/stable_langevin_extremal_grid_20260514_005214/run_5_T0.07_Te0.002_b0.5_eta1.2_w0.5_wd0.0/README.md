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
- temperature: `0.07`
- validation temperature: `0.002`
- stepsize: `1.2`
- nudged state-average samples: `16` every `1` full sweep(s)
- Minit/eval repeats: `8` / `24`
- workers: `8`

Best logged result:

- epoch: `2500`
- MSE: `1.1130513682288492`
- accuracy: `0.75`
- means: `[-0.3154797686987243, 0.9448133300700478, 0.5515013892228599, 0.9440784329464856]`

CSV: `timeavg_learning_metrics.csv`
Plot: `timeavg_learning_progress.png`
