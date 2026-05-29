# Local Paper MNIST ProcessManager

- architecture: `784 -> 784 -> 121 -> 40`
- radius: `6`
- workers: `32`
- batchsize: `32`
- train/test per class: `100` / `20`
- free/nudge reads: `3` / `3`
- free/nudge sweeps: `50` / `50`
- beta: `5.0`
- learning rates W0/W12/W2O/B: `0.004`, `0.004`, `0.004`, `0.0004`
- temperatures hot/cold/reverse: `5.0`, `0.01`, `1.0`
- gradient normalization: `mean`
- resumed from: `ext\IsingLearning\experiments\MNIST\single-hidden-local-28x28-to-11x11-readout\diagnostics\runs\r7_scheme_grid\mean_lr004_b32_traininternal_e30\r7\best_model.bin`
- worker graph adjacency: shared with source graph
- worker parameters: shared read-only during minibatch; source updates once after `FlushAtEnd()`
