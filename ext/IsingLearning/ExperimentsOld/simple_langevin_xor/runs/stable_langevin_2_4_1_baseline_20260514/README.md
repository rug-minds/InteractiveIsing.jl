# Simple 2->4->1 LocalLangevin XOR

This run compares normal EqProp against the split-snapshot variant on the smallest useful physical XOR graph: two input spins, four hidden spins, and one scalar output spin.

Both routes use unadjusted `LocalLangevin`, masked direct output clamping, trainable `Bilinear` weights and `MagField` biases, and no polynomial local potential.

- Temperature: `0.07`
- Stepsize: `0.8`
- Max drift fraction: `1.0`
- Free / early / nudged: `1200` / `20` / `1200`
- Minit / eval repeats: `1` / `24`
- Init mode: `random`

| Route | Best MSE | Best Acc |
|---|---:|---:|
| `normal` | 0.749794 | 0.75 |
| `split` | 0.766026 | 0.75 |

CSV: `simple_2_4_1_metrics.csv`
Plot: `simple_2_4_1_progress.png`
