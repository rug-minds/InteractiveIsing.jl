# Local CNN-Like XOR Grid

Static checkerboard input, two hidden layers, and configurable output readout.

- base architecture template: `8x8 -> HxH -> HxH -> 4x4`
- output mode: `pattern`
- output target: vertical pattern for XOR false and horizontal pattern for XOR true
- compared hidden sizes: `16`
- compared radii: `9`
- compared hidden periodicity: `false`
- epochs per config: `100`
- free/nudged sweeps: `2` / `2`
- workers: `32`
- repeats per case: `128`
- chunks per case: `8`
- snapshots every epochs: `10`
- optimizer: `adam`
- optimizer learning rate: `0.002`
- optimizer lr decay/min: `1.0` / `0.0`
- weight decay on couplings: `0.0`
- dynamics: `local`
- block size: `8`

## Best Results

| Rank | Config | Hidden | Radius | Periodic | Best Margin | Best Accuracy | First All-Correct | Best MSE |
|---:|---|---:|---:|---|---:|---:|---:|---:|
| 1 | `h16_r9_open` | 16 | 9 | false | -0.040535 | 0.75 | -1 | 0.929066 |

Metrics: `metrics.csv`
Summary: `summary.csv`
