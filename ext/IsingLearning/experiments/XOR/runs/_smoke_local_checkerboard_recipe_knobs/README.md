# Local CNN-Like XOR Grid

Static checkerboard input, two hidden layers, and majority-vote output.

- base architecture template: `8x8 -> HxH -> HxH -> 4x4`
- output target: all output spins `-1` for XOR false and `+1` for XOR true
- compared hidden sizes: `8`
- compared radii: `1`
- compared hidden periodicity: `false`
- epochs per config: `1`
- free/nudged sweeps: `1` / `1`
- workers: `2`
- repeats per case: `2`
- chunks per case: `1`
- snapshots every epochs: `1`
- optimizer: `adam`
- optimizer learning rate: `0.0015`
- weight decay on couplings: `0.0001`
- dynamics: `block`
- block size: `8`

## Best Results

| Rank | Config | Hidden | Radius | Periodic | Best Margin | Best Accuracy | First All-Correct | Best MSE |
|---:|---|---:|---:|---|---:|---:|---:|---:|
| 1 | `h8_r1_open` | 8 | 1 | false | -0.051113 | 0.5 | -1 | 1.028663 |

Plot: `learning_summary.png`
Metrics: `metrics.csv`
Summary: `summary.csv`
