# Simple 2->16->2 LocalLangevin XOR

This run compares normal EqProp against the split-snapshot variant on the smallest useful physical XOR graph: two input spins, four hidden spins, and one two-output spin.

Both routes use unadjusted `LocalLangevin`, masked direct output clamping, trainable `Bilinear` weights and `MagField` biases, and no polynomial local potential.

- Temperature: `0.02`
- Stepsize: `0.2`
- Max drift fraction: `1.0`
- Free / early / nudged: `1200` / `20` / `1200`
- Minit / eval repeats: `4` / `16`
- Init mode: random

| Route | Best MSE | Best Acc |
|---|---:|---:|
| `normal` | 0.316983 | 1.0 |

CSV: `simple_2_16_2_metrics.csv`
Plot: `simple_2_16_2_progress.png`
