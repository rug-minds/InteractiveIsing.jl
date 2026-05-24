# Edge Application XOR Grid

Static checkerboard evidence is applied on the left edge of a `16x16` propagation layer.
The output is read from an opposite edge line with the configured target code clamped during learning.

- output mode: `two_class`
- output target: replicated two-class edge code, first half false and second half true
- compared hidden NN values: `5,8`
- edge fanout: `1`
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

| Rank | Config | NN | Best Margin | Best Accuracy | First All-Correct | Best MSE |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `nn8` | 8 | 0.338141 | 1.0 | 20 | 0.22447 |
| 2 | `nn5` | 5 | 0.058146 | 1.0 | 20 | 0.6318 |

Plot: `learning_summary.png`
Metrics: `metrics.csv`
Summary: `summary.csv`
