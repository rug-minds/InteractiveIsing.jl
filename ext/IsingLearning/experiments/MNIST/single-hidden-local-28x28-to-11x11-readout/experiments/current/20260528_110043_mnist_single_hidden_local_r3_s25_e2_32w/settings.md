# Single-Hidden Local MNIST

- architecture: inactive input layer `784`, sampled layers `784 -> 121 -> 40`
- radius: `3`
- workers: `32`
- batchsize: `32`
- train/test per class: `100` / `20`
- free/nudge reads: `3` / `3`
- free/nudge sweeps: `25` / `25`
- beta: `5.0`
- optimizer: `adam`
- learning rates W0/W12/W2O/B: `0.004`, `0.004`, `0.004`, `0.0004`
- temperatures hot/cold/reverse: `5.0`, `0.01`, `1.0`
- gradient normalization: `mean`
- progress logging: `true`, every `10` indexed steps
- resumed from: `C:\Users\fenje\dev\InteractiveIsing.jl\ext\IsingLearning\experiments\MNIST\single-hidden-local-28x28-to-11x11-readout\diagnostics\runs\r7_scheme_grid\mean_lr004_b32_traininternal_e30\r7\best_model.bin`
- worker graph adjacency: shared with source graph
- worker parameters: shared read-only during minibatch; source updates once after `FlushAtEnd()`
