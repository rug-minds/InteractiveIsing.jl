# Simple 2->4->1 Time-Averaged XOR Learning

This run relearns the physical `2 -> 4 -> 1` XOR graph. It does not
reuse a learned graph. EqProp worker trajectories and validation
trajectories are launched and synchronized with `ProcessManager`.

Validation classifies by averaging the scalar output spin after a
burn-in period:

- burn-in full sweeps: `20`
- averaged samples: `50`
- sampling interval: one sample every `1` full sweep(s)

Training settings:

- epochs: `600`
- free/nudged relaxation: `600` / `600`
- beta: `2.0`
- temperature: `0.005`
- stepsize: `0.4`
- Minit/eval repeats: `4` / `8`
- workers: `8`

Best logged result:

- epoch: `400`
- MSE: `0.8897779281416303`
- accuracy: `0.75`
- means: `[-0.7178629851890923, 0.4298793970156324, 0.6875854142622586, 0.7483907006476463]`

CSV: `timeavg_learning_metrics.csv`
Plot: `timeavg_learning_progress.png`
