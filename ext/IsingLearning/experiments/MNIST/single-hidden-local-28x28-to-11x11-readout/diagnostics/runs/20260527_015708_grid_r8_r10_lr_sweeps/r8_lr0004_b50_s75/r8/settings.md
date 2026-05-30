# Single-Hidden Local MNIST

- architecture: inactive input layer `784`, sampled layers `784 -> 121 -> 40`
- radius: `8`
- workers: `8`
- batchsize: `32`
- train/test per class: `100` / `20`
- free/nudge reads: `3` / `3`
- free/nudge sweeps: `75` / `75`
- beta: `5.0`
- optimizer: `adam`
- learning rates W0/W12/W2O/B: `0.004`, `0.004`, `0.004`, `0.0004`
- temperatures hot/cold/reverse: `5.0`, `0.01`, `1.0`
- gradient normalization: `mean`
- progress logging: `false`, every `10` indexed steps
- worker graph adjacency: shared with source graph
- worker parameters: shared read-only during minibatch; source updates once after `FlushAtEnd()`
