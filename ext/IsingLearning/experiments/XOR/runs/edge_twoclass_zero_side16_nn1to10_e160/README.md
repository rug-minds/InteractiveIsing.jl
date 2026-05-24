# Edge Application XOR Grid

Static checkerboard evidence is applied on the left edge of a `16x16` propagation layer.
The output is read from an opposite edge line with the configured target code clamped during learning.

- output mode: `two_class`
- output target: replicated two-class edge code, first half false and second half true
- compared hidden NN values: `1,2,3,4,5,6,7,8,9,10`
- edge fanout: `1`
- epochs per config: `160`
- free/nudged sweeps: `20` / `20`
- workers: `32`
- repeats per case: `64`
- chunks per case: `8`
- snapshots every epochs: `40`
- optimizer: `adam`
- optimizer learning rate: `0.002`
- optimizer lr decay/min: `0.995` / `0.0002`
- weight decay on couplings: `0.0001`
- dynamics: `block`
- block size: `8`
- init mode: `zero`

## Best Results

| Rank | Config | NN | Best Margin | Best Accuracy | First All-Correct | Best MSE |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `nn7` | 7 | 0.864775 | 1.0 | 70 | 0.055691 |
| 2 | `nn9` | 9 | 0.851271 | 1.0 | 10 | 0.018841 |
| 3 | `nn6` | 6 | 0.775091 | 1.0 | 20 | 0.02829 |
| 4 | `nn8` | 8 | 0.674239 | 1.0 | 20 | 0.118954 |
| 5 | `nn5` | 5 | 0.321026 | 1.0 | 90 | 0.29787 |
| 6 | `nn10` | 10 | 0.20572 | 1.0 | 80 | 0.208996 |
| 7 | `nn2` | 2 | -0.003292 | 0.75 | -1 | 0.972784 |
| 8 | `nn3` | 3 | -0.017136 | 0.5 | -1 | 0.990372 |
| 9 | `nn4` | 4 | -0.086717 | 0.5 | -1 | 0.922336 |
| 10 | `nn1` | 1 | -0.121208 | 0.5 | -1 | 1.008644 |

Plot: `learning_summary.png`
Metrics: `metrics.csv`
Summary: `summary.csv`
