# Local CNN-Like XOR Grid

Static checkerboard input, two hidden layers, and majority-vote output.

- base architecture template: `8x8 -> HxH -> HxH -> 4x4`
- output target: all output spins `-1` for XOR false and `+1` for XOR true
- compared hidden sizes: `16`
- compared radii: `8,9`
- compared hidden periodicity: `false`
- epochs per config: `30`
- free/nudged sweeps: `5` / `5`
- workers: `32`
- repeats per case: `32`
- chunks per case: `8`
- snapshots every epochs: `10`
- optimizer: `adam`
- optimizer learning rate: `0.001`

## Best Results

| Rank | Config | Hidden | Radius | Periodic | Best Margin | Best Accuracy | First All-Correct | Best MSE |
|---:|---|---:|---:|---|---:|---:|---:|---:|
| 1 | `h16_r8_open` | 16 | 8 | false | -0.015371 | 0.75 | -1 | 0.926532 |
| 2 | `h16_r9_open` | 16 | 9 | false | -0.052793 | 0.5 | -1 | 0.921944 |

Plot: `learning_summary.png`
Metrics: `metrics.csv`
Summary: `summary.csv`
