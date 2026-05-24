# Local Paper MNIST ProcessManager

- architecture: `784 -> 784 -> 196 -> 40`
- radius: `3`
- workers: `32`
- batchsize: `32`
- train/test per class: `100` / `20`
- free/nudge reads: `3` / `3`
- free/nudge sweeps: `50` / `50`
- beta: `5.0`
- learning rates W0/W12/W2O/B: `0.048`, `0.048`, `0.048`, `0.016`
- temperatures hot/cold/reverse: `5.0`, `0.01`, `1.0`
- gradient normalization: `mean`
- worker graph adjacency: shared with source graph
- worker parameters: shared read-only during minibatch; source updates once after `FlushAtEnd()`
