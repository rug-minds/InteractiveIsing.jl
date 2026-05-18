# Simple 2->16->2 LocalLangevin XOR

This run compares normal EqProp against the split-snapshot variant on the smallest useful physical XOR graph: two input spins, four hidden spins, and one two-output spin.

Both routes use unadjusted `LocalLangevin`, masked direct output clamping, trainable `Bilinear` weights and `MagField` biases, and no polynomial local potential.

- Temperature: `0.005`
- Stepsize: `0.4`
- Max drift fraction: `1.0`
- Free / early / nudged: `600` / `20` / `600`
- Minit / eval repeats: `4` / `32`
- Init mode: `zero`

| Route | Best MSE | Best Acc |
|---|---:|---:|
| `normal` | 1.90067 | 0.75 |

CSV: `simple_2_16_2_metrics.csv`
Plot: `simple_2_16_2_progress.png`
