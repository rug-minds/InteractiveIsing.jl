# Local MNIST ProcessManager

- architecture: inactive input layer `784`, sampled layers `784 -> 121 -> 40`
- radius: `1`
- workers: `32`
- batchsize: `64`
- train/test per class: `100` / `20`
- free/nudge reads: `3` / `3`
- free/nudge sweeps: `50` / `50`
- beta: `5.0`
- optimizer: `adam`
- learning rates W0/W12/W2O/B: `0.003`, `0.003`, `0.003`, `0.0003`
- temperatures hot/cold/reverse: `5.0`, `0.01`, `1.0`
- gradient normalization: `sum`
- progress logging: `true`, every `1` indexed steps
- worker graph adjacency: shared with source graph
- worker parameters: shared read-only during minibatch; source updates once after `FlushAtEnd()`
