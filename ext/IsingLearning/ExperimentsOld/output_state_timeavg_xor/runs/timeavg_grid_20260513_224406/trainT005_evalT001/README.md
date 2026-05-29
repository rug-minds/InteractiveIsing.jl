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

- epochs: `8000`
- free/nudged relaxation: `300` / `300`
- beta: `2.0`
- temperature: `0.005`
- validation temperature: `0.001`
- stepsize: `0.4`
- nudged state-average samples: `20` every `1` full sweep(s)
- Minit/eval repeats: `8` / `16`
- workers: `8`

Best logged result:

- epoch: `6750`
- MSE: `0.13220850277179863`
- accuracy: `1.0`
- means: `[-0.6276793066044675, 0.7352705693363236, 0.7214811487211481, -0.5074993498689987]`

CSV: `timeavg_learning_metrics.csv`
Plot: `timeavg_learning_progress.png`
