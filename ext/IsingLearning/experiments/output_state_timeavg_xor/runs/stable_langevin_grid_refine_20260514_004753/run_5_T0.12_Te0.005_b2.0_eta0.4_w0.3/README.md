# Simple 2->4->1 Time-Averaged XOR Learning

This run relearns the physical `2 -> 4 -> 1` XOR graph. It does not
reuse a learned graph. EqProp worker trajectories and validation
trajectories are launched and synchronized with `ProcessManager`.

Validation classifies by averaging the scalar output spin after a
burn-in period:

- burn-in full sweeps: `30`
- averaged samples: `80`
- sampling interval: one sample every `1` full sweep(s)

Training settings:

- epochs: `1500`
- free/nudged relaxation: `220` / `220`
- beta: `2.0`
- temperature: `0.12`
- validation temperature: `0.005`
- stepsize: `0.4`
- nudged state-average samples: `12` every `1` full sweep(s)
- Minit/eval repeats: `6` / `16`
- workers: `8`

Best logged result:

- epoch: `1200`
- MSE: `1.0263420093527567`
- accuracy: `0.75`
- means: `[-0.8308622628393101, 0.9476412435273835, 0.4736392451519046, 0.9485798366132121]`

CSV: `timeavg_learning_metrics.csv`
Plot: `timeavg_learning_progress.png`
