# Local CNN-Like XOR Grid

Static checkerboard input, two hidden layers, and majority-vote output.

- base architecture template: `8x8 -> HxH -> HxH -> 4x4`
- output target: all output spins `-1` for XOR false and `+1` for XOR true
- compared hidden sizes: `8,16`
- compared radii: `8,9`
- compared hidden periodicity: `false`
- epochs per config: `40`
- free/nudged sweeps: `1` / `1`
- workers: `32`
- repeats per case: `64`
- chunks per case: `8`
- snapshots every epochs: `10`
- optimizer: `adam`
- optimizer learning rate: `0.005`
- weight decay on couplings: `0.0`
- dynamics: `block`
- block size: `8`

## Best Results

| Rank | Config | Hidden | Radius | Periodic | Best Margin | Best Accuracy | First All-Correct | Best MSE |
|---:|---|---:|---:|---|---:|---:|---:|---:|
| 1 | `h8_r8_open` | 8 | 8 | false | 0.00809 | 1.0 | 20 | 0.94471 |
| 2 | `h8_r9_open` | 8 | 9 | false | 0.005633 | 0.75 | 10 | 0.936281 |
| 3 | `h16_r9_open` | 16 | 9 | false | -0.007456 | 0.5 | -1 | 0.956851 |
| 4 | `h16_r8_open` | 16 | 8 | false | -0.008631 | 0.75 | -1 | 0.963539 |

Plot: `learning_summary.png`
Metrics: `metrics.csv`
Summary: `summary.csv`
