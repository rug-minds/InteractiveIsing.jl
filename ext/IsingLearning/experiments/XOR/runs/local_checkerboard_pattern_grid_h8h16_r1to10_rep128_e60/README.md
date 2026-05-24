# Local CNN-Like XOR Grid

Static checkerboard input, two hidden layers, and configurable output readout.

- base architecture template: `8x8 -> HxH -> HxH -> 4x4`
- output mode: `pattern`
- output target: vertical pattern for XOR false and horizontal pattern for XOR true
- compared hidden sizes: `8,16`
- compared radii: `1,2,3,4,5,6,7,8,9,10`
- compared hidden periodicity: `false`
- epochs per config: `60`
- free/nudged sweeps: `1` / `1`
- workers: `32`
- repeats per case: `128`
- chunks per case: `8`
- snapshots every epochs: `20`
- optimizer: `adam`
- optimizer learning rate: `0.005`
- optimizer lr decay/min: `0.98` / `0.0005`
- weight decay on couplings: `0.0001`
- dynamics: `block`
- block size: `8`

## Best Results

| Rank | Config | Hidden | Radius | Periodic | Best Margin | Best Accuracy | First All-Correct | Best MSE |
|---:|---|---:|---:|---|---:|---:|---:|---:|
| 1 | `h8_r2_open` | 8 | 2 | false | 0.045641 | 1.0 | 60 | 0.880403 |
| 2 | `h16_r5_open` | 16 | 5 | false | 0.045588 | 1.0 | 60 | 0.857864 |
| 3 | `h8_r7_open` | 8 | 7 | false | 0.03679 | 1.0 | 10 | 0.903595 |
| 4 | `h16_r2_open` | 16 | 2 | false | 0.012278 | 1.0 | 10 | 0.951481 |
| 5 | `h16_r4_open` | 16 | 4 | false | 0.010787 | 1.0 | 40 | 0.90688 |
| 6 | `h8_r6_open` | 8 | 6 | false | 0.010351 | 1.0 | 20 | 0.897376 |
| 7 | `h8_r10_open` | 8 | 10 | false | 0.010029 | 1.0 | 40 | 0.943497 |
| 8 | `h8_r1_open` | 8 | 1 | false | 0.00446 | 1.0 | 20 | 0.92274 |
| 9 | `h16_r9_open` | 16 | 9 | false | 0.000698 | 1.0 | 60 | 0.940451 |
| 10 | `h8_r8_open` | 8 | 8 | false | -0.000736 | 0.5 | -1 | 0.945138 |
| 11 | `h16_r8_open` | 16 | 8 | false | -0.002993 | 0.5 | -1 | 0.94189 |
| 12 | `h16_r7_open` | 16 | 7 | false | -0.003347 | 0.75 | -1 | 0.906808 |
| 13 | `h16_r6_open` | 16 | 6 | false | -0.005469 | 0.75 | -1 | 0.889739 |
| 14 | `h8_r3_open` | 8 | 3 | false | -0.007974 | 0.5 | -1 | 0.956233 |
| 15 | `h16_r10_open` | 16 | 10 | false | -0.017711 | 0.75 | -1 | 0.954107 |
| 16 | `h16_r1_open` | 16 | 1 | false | -0.028499 | 0.5 | -1 | 0.952834 |
| 17 | `h8_r4_open` | 8 | 4 | false | -0.029309 | 0.75 | -1 | 0.941974 |
| 18 | `h8_r9_open` | 8 | 9 | false | -0.031463 | 0.5 | -1 | 0.991089 |
| 19 | `h8_r5_open` | 8 | 5 | false | -0.042986 | 0.75 | -1 | 0.954113 |
| 20 | `h16_r3_open` | 16 | 3 | false | -0.052591 | 0.75 | -1 | 1.003829 |

Metrics: `metrics.csv`
Summary: `summary.csv`
