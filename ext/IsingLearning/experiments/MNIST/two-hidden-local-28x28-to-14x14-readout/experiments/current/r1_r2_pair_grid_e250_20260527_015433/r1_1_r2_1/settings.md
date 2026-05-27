# Two-Hidden Local MNIST

- architecture: inactive input layer `784`, sampled layers `121 -> 25 -> 40`
- radii r1/r2: `1` / `1`
- workers: `32`
- batchsize: `32`
- train/test per class: `100` / `20`
- free/nudge reads: `3` / `3`
- free/nudge sweeps: `75` / `75`
- beta: `5.0`
- optimizer: `adam`
- learning rates W0/W12/W2O/B: `0.012`, `0.012`, `0.012`, `0.0012`
- temperatures hot/cold/reverse: `5.0`, `0.01`, `1.0`
- gradient normalization: `mean`
- progress logging: `true`, every `10` indexed steps
- worker graph adjacency: shared with source graph
- worker parameters: shared read-only during minibatch; source updates once after `FlushAtEnd()`
