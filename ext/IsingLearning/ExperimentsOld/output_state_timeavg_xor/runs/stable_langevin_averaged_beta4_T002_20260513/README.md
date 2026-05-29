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

- epochs: `1200`
- free/nudged relaxation: `300` / `300`
- beta: `4.0`
- temperature: `0.002`
- validation temperature: `0.0002`
- stepsize: `0.4`
- nudged state-average samples: `20` every `1` full sweep(s)
- Minit/eval repeats: `8` / `16`
- workers: `8`

Best logged result:

- epoch: `1100`
- MSE: `1.0657250118262578`
- accuracy: `0.75`
- means: `[-0.8019562077512298, 0.602642160413727, 0.673082964588894, 0.989701159203085]`

CSV: `timeavg_learning_metrics.csv`
Plot: `timeavg_learning_progress.png`
