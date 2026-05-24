# Edge Application XOR Grid

Static checkerboard evidence is applied on the left edge of a `8x8` propagation layer.
The output is read from an opposite edge line with the configured target code clamped during learning.

- output mode: `two_class`
- output target: replicated two-class edge code, first half false and second half true
- compared hidden NN values: `5`
- edge fanout: `1`
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

| Rank | Config | NN | Best Margin | Best Accuracy | First All-Correct | Best MSE |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `nn5` | 5 | -0.011242 | 0.5 | -1 | 0.995426 |

Metrics: `metrics.csv`
Summary: `summary.csv`
