# Local CNN-Like XOR Grid

Static checkerboard input, two hidden layers, and configurable output readout.

- base architecture template: `8x8 -> HxH -> HxH -> 4x4`
- output mode: `two_class`
- output target: replicated two-class code, first half false and second half true
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
- optimizer lr decay/min: `1.0` / `0.0`
- weight decay on couplings: `0.0`
- dynamics: `block`
- block size: `8`
- init mode: `random`

## Best Results

| Rank | Config | Hidden | Radius | Periodic | Best Margin | Best Accuracy | First All-Correct | Best MSE |
|---:|---|---:|---:|---|---:|---:|---:|---:|
| 1 | `h8_r1_open` | 8 | 1 | false | -0.21665 | 0.75 | -1 | 1.04585 |

Metrics: `metrics.csv`
Summary: `summary.csv`
