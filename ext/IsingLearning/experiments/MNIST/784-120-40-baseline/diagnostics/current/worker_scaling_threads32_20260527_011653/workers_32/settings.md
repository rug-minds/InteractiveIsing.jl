# MNIST 784-120-40 Adam Baseline

- paper: https://arxiv.org/pdf/2305.18321
- architecture: `784 -> 120 -> 40`
- input handling: MNIST pixels in `[0, 1]` are folded into a worker-local second `MagField`
- workers: `32`
- worker graph adjacency: pointer-shared with source graph
- learning step: one-sided free/nudged Process `LoopAlgorithm` from `@Routine`
- optimiser: `Optimisers.Adam(0.003)`
- epochs/batchsize: `1` / `128`
- train/test per class: `500` / `100`
- train eval per class: `20`
- sweeps/relaxation steps: `100.0` / `16000`
- beta/temp/stepsize: `5.0` / `0.001` / `0.5`
- weight scale/decay: `0.005` / `0.0`
