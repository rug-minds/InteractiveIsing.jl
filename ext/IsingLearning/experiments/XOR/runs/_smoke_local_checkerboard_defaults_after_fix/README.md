# Local CNN-Like XOR Grid

Static checkerboard input, two hidden layers, and configurable output readout.

- base architecture template: `8x8 -> HxH -> HxH -> 4x4`
- output mode: `two_class`
- output target: replicated two-class code, first half false and second half true
- compared hidden sizes: `8`
- compared radii: `8`
- compared hidden periodicity: `false`
- epochs per config: `1`
- free/nudged sweeps: `1` / `1`
- workers: `2`
- repeats per case: `2`
- chunks per case: `1`
- snapshots every epochs: `1`
- optimizer: `adam`
- optimizer learning rate: `0.002`
- optimizer lr decay/min: `0.995` / `0.0002`
- weight decay on couplings: `0.0001`
- dynamics: `block`
- block size: `8`
- init mode: `zero`

## Best Results

| Rank | Config | Hidden | Radius | Periodic | Best Margin | Best Accuracy | First All-Correct | Best MSE |
|---:|---|---:|---:|---|---:|---:|---:|---:|
| 1 | `h8_r8_open` | 8 | 8 | false | -0.005353 | 0.5 | -1 | 0.994465 |

Metrics: `metrics.csv`
Summary: `summary.csv`
