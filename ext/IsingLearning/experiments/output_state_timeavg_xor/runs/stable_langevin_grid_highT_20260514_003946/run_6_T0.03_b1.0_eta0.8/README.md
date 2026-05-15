# Simple 2->4->1 Time-Averaged XOR Learning

This run relearns the physical `2 -> 4 -> 1` XOR graph. It does not
reuse a learned graph. EqProp worker trajectories and validation
trajectories are launched and synchronized with `ProcessManager`.

Validation classifies by averaging the scalar output spin after a
burn-in period:

- burn-in full sweeps: `20`
- averaged samples: `40`
- sampling interval: one sample every `1` full sweep(s)

Training settings:

- epochs: `500`
- free/nudged relaxation: `180` / `180`
- beta: `1.0`
- temperature: `0.03`
- validation temperature: `0.03`
- stepsize: `0.8`
- nudged state-average samples: `10` every `1` full sweep(s)
- Minit/eval repeats: `4` / `8`
- workers: `8`

Best logged result:

- epoch: `300`
- MSE: `0.8455795222409443`
- accuracy: `0.75`
- means: `[-0.7523451048380319, 0.8407341754218091, 0.8196849820656367, 0.8064069401147824]`

CSV: `timeavg_learning_metrics.csv`
Plot: `timeavg_learning_progress.png`
