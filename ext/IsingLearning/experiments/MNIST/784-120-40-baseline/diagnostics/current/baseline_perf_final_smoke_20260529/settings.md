# MNIST 784-120-40 Adam Baseline

- paper: https://arxiv.org/pdf/2305.18321
- architecture: `784 -> 120 -> 40`
- input handling: MNIST pixels in `[0, 1]` are folded into worker-local field state
- workers: `2`
- worker graph adjacency: pointer-shared with source graph
- learning step: one-sided free/nudged Process `LoopAlgorithm` from `@Routine`
- validation: ProcessManager with preallocated jobs and worker-local stats
- job buffers: preallocated train/eval job vectors reused across minibatches/evaluations
- optimiser: `Optimisers.Adam(0.003)`
- epochs/batchsize: `0` / `2`
- train/test per class: `1` / `1`
- train eval per class: `0`
- sweeps/relaxation steps: `0.001` / `1`
- beta/temp/stepsize: `5.0` / `0.001` / `0.5`
- weight scale/decay: `0.005` / `0.0`
- resume from: `none`
- resume epoch: `-1`
