# Simple 2->4->1 Time-Averaged XOR Learning

This run relearns the physical `2 -> 4 -> 1` XOR graph. It does not
reuse a learned graph. EqProp worker trajectories and validation
trajectories are launched and synchronized with `ProcessManager`.

Validation classifies by averaging the scalar output spin after a
burn-in period:

- burn-in full sweeps: `2`
- averaged samples: `5`
- sampling interval: one sample every `1` full sweep(s)

Training settings:

- epochs: `1`
- free/nudged relaxation: `10` / `10`
- beta: `0.2`
- temperature: `0.01`
- stepsize: `0.2`
- Minit/eval repeats: `1` / `1`
- workers: `2`

Best logged result:

- epoch: `1`
- MSE: `0.6307619831281982`
- accuracy: `0.75`
- means: `[-0.03285541040143179, 0.33310857256373694, -0.06644896341713055, -0.9250219514687059]`

CSV: `timeavg_learning_metrics.csv`
Plot: `timeavg_learning_progress.png`
