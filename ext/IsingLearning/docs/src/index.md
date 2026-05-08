# IsingLearning

`IsingLearning` is the learning extension for `InteractiveIsing`. It contains
the active equilibrium-propagation training path, graph factories for layered
Boltzmann-style models, and small runnable learning examples.

## Active Training Path

The currently used threaded path is:

```text
init_mnist_trainer
-> _worker_process
-> Forward_and_Nudged
-> contrastive_gradient
-> Optimisers.update
```

The older experimental files `ComputeGradients.jl`, `ChainRules.jl`, and
`ep_train_step!` are not part of this active path.

## Important Files

- `src/GraphSetup.jl`: graph factories such as `ReducedBoltzmannArchitecture`.
- `src/Dynamics.jl`: forward and nudged relaxation routines.
- `src/Gradient.jl`: symmetric EP gradient estimator used by the active path.
- `src/ThreadedMNISTLoop.jl`: trainer, worker process setup, batching, and evaluation.
- `src/ReadoutClamping.jl`: extension-only clamping term for scalar linear readouts.

## Examples

- `examples/tiny_mnist.jl`: tiny MNIST smoke test on the active threaded path.
- `examples/xor_statistical_ep.jl`: statistical XOR experiment using repeated relaxations.
- `examples/gradient_trace.jl`: gradient tracing/debugging.

## Notes

Development findings from the XOR debugging runs are tracked in
[`xor_findings.md`](xor_findings.md).

The first successful statistical XOR run is documented in
[`xor_successful_learning.md`](xor_successful_learning.md).

The distinction between exact equilibrium propagation and the successful
short-run contrastive update is explained in
[`xor_short_run_contrast.md`](xor_short_run_contrast.md).
