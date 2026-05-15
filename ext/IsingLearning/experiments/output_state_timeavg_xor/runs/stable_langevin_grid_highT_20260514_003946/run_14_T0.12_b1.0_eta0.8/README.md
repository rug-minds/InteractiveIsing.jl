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
- temperature: `0.12`
- validation temperature: `0.12`
- stepsize: `0.8`
- nudged state-average samples: `10` every `1` full sweep(s)
- Minit/eval repeats: `4` / `8`
- workers: `8`

Best logged result:

- epoch: `500`
- MSE: `0.8582285483440592`
- accuracy: `0.75`
- means: `[-0.6383402592003006, -0.5498788447153128, 0.48631923839253555, -0.2024261368881042]`

CSV: `timeavg_learning_metrics.csv`
Plot: `timeavg_learning_progress.png`
