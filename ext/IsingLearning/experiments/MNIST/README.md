# MNIST Experiments

The training files default to `32` `ProcessManager` workers. Start Julia with
`-t 32` for the intended run shape.

Clean MNIST experiment files live here. The exploratory MNIST work remains
under:

```text
ext/IsingLearning/ExperimentsOld/mnist_manager
```

## Files

- `mnist_local_nn_grid.jl`: manager-backed local MNIST sweep. It compares
  square inter-layer nearest-neighbor radii for
  `28x28 -> H1 -> H2 -> 10 * output_replicas`, writes aggregate CSVs, saves
  final parameters per configuration, and produces `mnist_local_nn_summary.png`.
  Defaults use a bounded `1024` train / `256` validation subset so the sweep is
  not accidentally a full-data multi-radius run.
- `plot_paper_nn_grid.jl`: aggregates and plots the paper-style local MNIST
  radius runs that use the older working one-sided EP recipe.
- `plot_paper_nn_hyper_grid.jl`: combines several paper-style radius sweeps
  with different temperature and learning hyperparameters.
- `plot_current_results.jl`: post-processes all recognized CSV files under
  `runs/current` and writes PNG learning/evaluation plots beside them.
- `mnist_local_paper_manager_grid.jl`: ProcessManager-backed version of the
  working paper-style local MNIST recipe. Workers share the sampled graph
  adjacency and read shared parameters during a minibatch, accumulate local
  paper-style gradients, and update the source parameters after `FlushAtEnd()`.
- `mnist_cnn_two_layer_nn_grid.jl`: wrapper around the clean paper-manager
  recipe for two-hidden-layer CNN-style MNIST sweeps. It varies hidden2 size
  and local NN/fanout radius, writes aggregate CSVs, and plots the learning
  curves.
- `evaluate_paper_manager_checkpoint.jl`: evaluates a saved
  `mnist_local_paper_manager_grid.jl` checkpoint on a balanced test slice with
  configurable free reads/sweeps.
- `mnist_inlaid_input_diagnostics.jl`: diagnostics for the 55x55 inlaid-input
  architecture where fixed MNIST pixels are separated by live spins.
- `mnist_inlaid_input_training.jl`: ProcessManager-backed trainer for the
  inlaid-input architecture. The first useful recipe trains readout couplings
  from fixed `[0, 1]` pixel sites to 40 output replicas, keeps separator sites
  live, uses output competition, and writes metrics/plots/checkpoints per run.

Run from the repository root:

```powershell
julia -t 32 --project=ext/IsingLearning ext/IsingLearning/experiments/MNIST/mnist_local_nn_grid.jl
julia -t 32 --project=ext/IsingLearning ext/IsingLearning/experiments/MNIST/mnist_local_paper_manager_grid.jl
julia -t 32 --project=ext/IsingLearning ext/IsingLearning/experiments/MNIST/mnist_cnn_two_layer_nn_grid.jl
julia -t 32 --project=ext/IsingLearning ext/IsingLearning/experiments/MNIST/evaluate_paper_manager_checkpoint.jl
julia -t 32 --project=ext/IsingLearning ext/IsingLearning/experiments/MNIST/mnist_inlaid_input_diagnostics.jl
julia -t 32 --project=ext/IsingLearning ext/IsingLearning/experiments/MNIST/mnist_inlaid_input_training.jl
julia --project=ext/IsingLearning ext/IsingLearning/experiments/MNIST/plot_current_results.jl
julia --project=ext/IsingLearning ext/IsingLearning/experiments/MNIST/plot_paper_nn_grid.jl
julia --project=ext/IsingLearning ext/IsingLearning/experiments/MNIST/plot_paper_nn_hyper_grid.jl
```

Set `ISING_MNIST_LOCAL_NN_TRAIN_LIMIT=""` to run on the full training split.

Useful quick smoke settings:

```powershell
$env:ISING_MNIST_LOCAL_NN_RADII="1,3"
$env:ISING_MNIST_LOCAL_NN_EPOCHS="1"
$env:ISING_MNIST_LOCAL_NN_TRAIN_LIMIT="16"
$env:ISING_MNIST_LOCAL_NN_VALIDATION_LIMIT="16"
$env:ISING_MNIST_LOCAL_NN_BATCHSIZE="4"
$env:ISING_MNIST_LOCAL_NN_SWEEPS="1"
julia -t 32 --project=ext/IsingLearning ext/IsingLearning/experiments/MNIST/mnist_local_nn_grid.jl
```
