# MNIST 784-120-40 Adam Baseline

- paper: https://arxiv.org/pdf/2305.18321
- architecture: `784 -> 120 -> 40`
- input handling: MNIST pixels in `[0, 1]` are folded into a worker-local second `MagField`
- workers: `1`
- worker graph adjacency: pointer-shared with source graph
- learning step: one-sided free/nudged Process `LoopAlgorithm` from `@Routine`
- optimiser: `Optimisers.Adam(0.003)`
- epochs/batchsize: `1` / `10`
- train/test per class: `1` / `1`
- train eval per class: `1`
- sweeps/relaxation steps: `1.0` / `160`
- beta/temp/stepsize: `5.0` / `0.001` / `0.5`
- weight scale/decay: `0.005` / `0.0`
