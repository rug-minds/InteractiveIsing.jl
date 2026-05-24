# Local CNN-Like XOR Grid

Static checkerboard input, two hidden layers, and configurable output readout.

- base architecture template: `8x8 -> HxH -> HxH -> 4x4`
- output mode: `two_class`
- output target: replicated two-class code, first half false and second half true
- compared hidden sizes: `8`
- compared radii: `8,9`
- compared hidden periodicity: `false`
- epochs per config: `80`
- free/nudged sweeps: `20` / `20`
- workers: `32`
- repeats per case: `64`
- chunks per case: `8`
- snapshots every epochs: `20`
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
| 1 | `h8_r8_open` | 8 | 8 | false | 0.635849 | 1.0 | 25 | 0.044284 |
| 2 | `h8_r9_open` | 8 | 9 | false | -0.035646 | 0.75 | -1 | 0.336362 |

Metrics: `metrics.csv`
Summary: `summary.csv`
Plot: `learning_summary.png`
High-repeat validation: `validation_h8_r8_512.md`
