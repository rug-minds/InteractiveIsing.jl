# Local CNN-Like XOR Grid

Static checkerboard input, two hidden layers, and configurable output readout.

- base architecture template: `8x8 -> HxH -> HxH -> 4x4`
- output mode: `pattern`
- output target: vertical pattern for XOR false and horizontal pattern for XOR true
- compared hidden sizes: `8,16`
- compared radii: `2,5`
- compared hidden periodicity: `false`
- epochs per config: `80`
- free/nudged sweeps: `1` / `1`
- workers: `32`
- repeats per case: `128`
- chunks per case: `8`
- snapshots every epochs: `20`
- optimizer: `adam`
- optimizer learning rate: `0.002`
- optimizer lr decay/min: `0.98` / `0.0002`
- weight decay on couplings: `0.0001`
- dynamics: `block`
- block size: `8`
- init mode: `random`

## Best Results

| Rank | Config | Hidden | Radius | Periodic | Best Margin | Best Accuracy | First All-Correct | Best MSE |
|---:|---|---:|---:|---|---:|---:|---:|---:|
| 1 | `h8_r5_open` | 8 | 5 | false | 0.0202 | 1.0 | 40 | 0.927077 |
| 2 | `h8_r2_open` | 8 | 2 | false | 0.013476 | 1.0 | 10 | 0.943121 |
| 3 | `h16_r5_open` | 16 | 5 | false | 0.009113 | 0.75 | 50 | 0.929648 |
| 4 | `h16_r2_open` | 16 | 2 | false | -0.000295 | 0.75 | -1 | 0.93868 |

Metrics: `metrics.csv`
Summary: `summary.csv`
