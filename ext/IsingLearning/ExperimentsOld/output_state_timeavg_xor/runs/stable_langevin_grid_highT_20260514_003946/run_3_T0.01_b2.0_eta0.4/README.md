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
- beta: `2.0`
- temperature: `0.01`
- validation temperature: `0.01`
- stepsize: `0.4`
- nudged state-average samples: `10` every `1` full sweep(s)
- Minit/eval repeats: `4` / `8`
- workers: `8`

Best logged result:

- epoch: `400`
- MSE: `0.9757887381361142`
- accuracy: `0.75`
- means: `[-0.6777821482556606, 0.9329300746246522, 0.6994873529002872, 0.9247141041216975]`

CSV: `timeavg_learning_metrics.csv`
Plot: `timeavg_learning_progress.png`
