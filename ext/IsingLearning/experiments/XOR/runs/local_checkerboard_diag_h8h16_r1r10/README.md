# Local CNN-Like XOR Grid

Static checkerboard input, two hidden layers, and majority-vote output.

- base architecture template: `8x8 -> HxH -> HxH -> 4x4`
- output target: all output spins `-1` for XOR false and `+1` for XOR true
- compared hidden sizes: `8,16`
- compared radii: `1,10`
- compared hidden periodicity: `false`
- epochs per config: `2`
- free/nudged sweeps: `5` / `5`
- workers: `32`
- repeats per case: `32`
- chunks per case: `8`
- snapshots every epochs: `1`
- optimizer: `adam`
- optimizer learning rate: `0.0015`

## Best Results

| Rank | Config | Hidden | Radius | Periodic | Best Margin | Best Accuracy | First All-Correct | Best MSE |
|---:|---|---:|---:|---|---:|---:|---:|---:|
| 1 | `h16_r1_open` | 16 | 1 | false | -0.012137 | 0.75 | -1 | 0.962484 |
| 2 | `h16_r10_open` | 16 | 10 | false | -0.028126 | 0.25 | -1 | 1.00727 |
| 3 | `h8_r1_open` | 8 | 1 | false | -0.048677 | 0.5 | -1 | 0.923203 |
| 4 | `h8_r10_open` | 8 | 10 | false | -0.057608 | 0.5 | -1 | 0.96121 |

Plot: `learning_summary.png`
Metrics: `metrics.csv`
Summary: `summary.csv`
