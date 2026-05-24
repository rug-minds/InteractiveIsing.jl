# Local CNN-Like XOR Grid

Static checkerboard input, two hidden layers, and majority-vote output.

- base architecture template: `8x8 -> HxH -> HxH -> 4x4`
- output target: all output spins `-1` for XOR false and `+1` for XOR true
- compared hidden sizes: `8,16`
- compared radii: `1,2,3,4,5,6,7,8,9,10`
- compared hidden periodicity: `false`
- epochs per config: `10`
- free/nudged sweeps: `5` / `5`
- workers: `32`
- repeats per case: `32`
- chunks per case: `8`
- snapshots every epochs: `10`
- optimizer: `adam`
- optimizer learning rate: `0.0015`

## Best Results

| Rank | Config | Hidden | Radius | Periodic | Best Margin | Best Accuracy | First All-Correct | Best MSE |
|---:|---|---:|---:|---|---:|---:|---:|---:|
| 1 | `h16_r8_open` | 16 | 8 | false | 0.013796 | 1.0 | 0 | 0.931955 |
| 2 | `h16_r9_open` | 16 | 9 | false | 0.010259 | 1.0 | 5 | 0.909067 |
| 3 | `h8_r8_open` | 8 | 8 | false | 0.003844 | 1.0 | 5 | 0.912173 |
| 4 | `h8_r2_open` | 8 | 2 | false | -0.014551 | 0.5 | -1 | 0.93422 |
| 5 | `h8_r4_open` | 8 | 4 | false | -0.01612 | 0.75 | -1 | 0.951354 |
| 6 | `h16_r3_open` | 16 | 3 | false | -0.016569 | 0.75 | -1 | 0.958396 |
| 7 | `h16_r5_open` | 16 | 5 | false | -0.023062 | 0.5 | -1 | 1.00442 |
| 8 | `h8_r9_open` | 8 | 9 | false | -0.024252 | 0.25 | -1 | 1.002203 |
| 9 | `h8_r10_open` | 8 | 10 | false | -0.032851 | 0.75 | -1 | 0.946183 |
| 10 | `h16_r10_open` | 16 | 10 | false | -0.034224 | 0.75 | -1 | 0.946833 |
| 11 | `h8_r7_open` | 8 | 7 | false | -0.044543 | 0.75 | -1 | 0.932631 |
| 12 | `h16_r2_open` | 16 | 2 | false | -0.049191 | 0.75 | -1 | 0.984366 |
| 13 | `h16_r1_open` | 16 | 1 | false | -0.051543 | 0.5 | -1 | 0.97455 |
| 14 | `h8_r1_open` | 8 | 1 | false | -0.051627 | 0.25 | -1 | 1.024045 |
| 15 | `h8_r5_open` | 8 | 5 | false | -0.053109 | 0.5 | -1 | 0.975173 |
| 16 | `h16_r4_open` | 16 | 4 | false | -0.05991 | 0.5 | -1 | 1.011848 |
| 17 | `h8_r6_open` | 8 | 6 | false | -0.060259 | 0.5 | -1 | 0.98125 |
| 18 | `h8_r3_open` | 8 | 3 | false | -0.067713 | 0.75 | -1 | 0.927163 |
| 19 | `h16_r7_open` | 16 | 7 | false | -0.081101 | 0.75 | -1 | 0.969025 |
| 20 | `h16_r6_open` | 16 | 6 | false | -0.110446 | 0.75 | -1 | 1.041417 |

Plot: `learning_summary.png`
Metrics: `metrics.csv`
Summary: `summary.csv`
