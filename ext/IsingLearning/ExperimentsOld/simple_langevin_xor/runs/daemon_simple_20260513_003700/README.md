# Simple 2->4->1 LocalLangevin XOR

This run compares normal EqProp against the split-snapshot variant on the smallest useful physical XOR graph: two input spins, four hidden spins, and one scalar output spin.

Both routes use unadjusted `LocalLangevin`, masked direct output clamping, trainable `Bilinear` weights and `MagField` biases, and no polynomial local potential.

- Temperature: `0.05`
- Stepsize: `0.4`
- Max drift fraction: `0.6`
- Free / early / nudged: `600` / `20` / `600`
- Minit / eval repeats: `8` / `16`
- Init mode: `random`

| Route | Best MSE | Best Acc |
|---|---:|---:|
| `normal` | 0.819531 | 0.75 |
| `split` | 0.816933 | 0.75 |

CSV: `simple_2_4_1_metrics.csv`
Plot: `simple_2_4_1_progress.png`
