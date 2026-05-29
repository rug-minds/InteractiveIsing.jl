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

- epochs: `1`
- free/nudged relaxation: `300` / `300`
- beta: `2.0`
- temperature: `0.005`
- validation temperature: `0.005`
- stepsize: `0.4`
- nudged state-average samples: `1` every `1` full sweep(s)
- Minit/eval repeats: `8` / `16`
- workers: `2`

Best logged result:

- epoch: `0`
- MSE: `1.437915278380515`
- accuracy: `0.5`
- means: `[-0.7841687277943373, -0.8708688445920878, 0.5732204688349172, 0.42224715628941356]`

CSV: `timeavg_learning_metrics.csv`
Plot: `timeavg_learning_progress.png`
