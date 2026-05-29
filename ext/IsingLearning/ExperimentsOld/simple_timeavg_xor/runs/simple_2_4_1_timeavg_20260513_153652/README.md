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

- epochs: `800`
- free/nudged relaxation: `600` / `600`
- beta: `2.0`
- temperature: `0.005`
- stepsize: `0.4`
- nudged derivative samples: `20` every `1` full sweep(s)
- Minit/eval repeats: `4` / `8`
- workers: `8`

Best logged result:

- epoch: `800`
- MSE: `0.8733348935116023`
- accuracy: `0.75`
- means: `[-0.6400601041053471, 0.061899424778275036, 0.6834108491234612, 0.5438657537835654]`

CSV: `timeavg_learning_metrics.csv`
Plot: `timeavg_learning_progress.png`
