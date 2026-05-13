# Simple 2->4->1 LocalLangevin XOR

This run compares normal EqProp against the split-snapshot variant on the smallest useful physical XOR graph: two input spins, four hidden spins, and one scalar output spin.

Both routes use unadjusted `LocalLangevin`, masked direct output clamping, trainable `Bilinear` weights and `MagField` biases, and no polynomial local potential.

- Temperature: `0.02`
- Stepsize: `0.2`
- Max drift fraction: `0.6`
- Free / early / nudged: `150` / `20` / `150`
- Minit / eval repeats: `1` / `2`

| Route | Best MSE | Best Acc |
|---|---:|---:|
| `normal` | 0.72212 | 0.5 |
| `split` | 0.891148 | 0.5 |

CSV: `simple_2_4_1_metrics.csv`
Plot: `simple_2_4_1_progress.png`
