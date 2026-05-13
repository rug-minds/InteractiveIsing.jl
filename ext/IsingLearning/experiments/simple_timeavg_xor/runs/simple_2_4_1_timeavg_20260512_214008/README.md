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

- epochs: `5`
- free/nudged relaxation: `10` / `10`
- beta: `0.2`
- temperature: `0.01`
- stepsize: `0.2`
- Minit/eval repeats: `1` / `1`
- workers: `2`

Best logged result:

- epoch: `5`
- MSE: `0.796407468420315`
- accuracy: `0.75`
- means: `[-0.06515645310088203, 0.00917817270526427, -0.12973737354076606, -0.768347264458481]`

CSV: `timeavg_learning_metrics.csv`
Plot: `timeavg_learning_progress.png`
