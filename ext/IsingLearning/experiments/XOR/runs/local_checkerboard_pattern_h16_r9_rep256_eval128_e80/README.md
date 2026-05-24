# Local CNN-Like XOR Grid

Static checkerboard input, two hidden layers, and configurable output readout.

- base architecture template: `8x8 -> HxH -> HxH -> 4x4`
- output mode: `pattern`
- output target: vertical pattern for XOR false and horizontal pattern for XOR true
- compared hidden sizes: `16`
- compared radii: `9`
- compared hidden periodicity: `false`
- epochs per config: `80`
- free/nudged sweeps: `1` / `1`
- workers: `32`
- repeats per case: `256`
- chunks per case: `8`
- snapshots every epochs: `10`
- optimizer: `adam`
- optimizer learning rate: `0.005`
- optimizer lr decay/min: `0.98` / `0.0005`
- weight decay on couplings: `0.0001`
- dynamics: `block`
- block size: `8`

## Best Results

| Rank | Config | Hidden | Radius | Periodic | Best Margin | Best Accuracy | First All-Correct | Best MSE |
|---:|---|---:|---:|---|---:|---:|---:|---:|
| 1 | `h16_r9_open` | 16 | 9 | false | -0.007618 | 0.75 | -1 | 0.967097 |

Metrics: `metrics.csv`
Summary: `summary.csv`
