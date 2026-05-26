# Local MNIST ProcessManager

- architecture: `784 -> 784 -> 121 -> 40`
- radius: `7`
- workers: `2`
- batchsize: `2`
- train/test per class: `1` / `1`
- free/nudge reads: `1` / `1`
- free/nudge sweeps: `2` / `2`
- beta: `5.0`
- learning rates W0/W12/W2O/B: `0.003`, `0.003`, `0.003`, `0.0003`
- temperatures hot/cold/reverse: `5.0`, `0.01`, `1.0`
- gradient normalization: `sum`
- worker graph adjacency: shared with source graph
- worker parameters: shared read-only during minibatch; source updates once after `FlushAtEnd()`
