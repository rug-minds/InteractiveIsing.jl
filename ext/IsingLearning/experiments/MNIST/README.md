# MNIST Experiments

The MNIST experiment tree is organized by architecture. Each architecture folder
contains its runnable scripts, `schematic.png`, aggregate plots, and retained
successful runs under `experiments/current/<experiment-name>`.

## Architecture folders

- `single-hidden-local-28x28-to-11x11-readout`: local manager baseline with
  `28x28` input fields, an `11x11` hidden layer, and `40` output spins.
- `two-hidden-local-28x28-to-14x14-readout`: CNN-style local manager runs with
  `28x28` input fields, a `28x28` first hidden layer, a `14x14` second hidden
  layer, and `40` output spins.
- `inlaid-55x55-pixel-readout`: inlaid-input architecture with fixed MNIST
  pixels placed in a `55x55` layer and read out to `40` output spins.
- `diagnostics`: timing, relaxation, and failed-but-informative probes. These
  are not mixed into the retained learning experiments.

Per-experiment plots live beside the CSV/checkpoint files they summarize.
Aggregate plots live in each architecture folder under `aggregate_plots`.

## Commands

Run from the repository root:

```powershell
julia -t 32 --project=ext/IsingLearning ext/IsingLearning/experiments/MNIST/single-hidden-local-28x28-to-11x11-readout/mnist_local_paper_manager_grid.jl
julia -t 32 --project=ext/IsingLearning ext/IsingLearning/experiments/MNIST/single-hidden-local-28x28-to-11x11-readout/evaluate_paper_manager_checkpoint.jl
julia -t 32 --project=ext/IsingLearning ext/IsingLearning/experiments/MNIST/two-hidden-local-28x28-to-14x14-readout/mnist_cnn_two_layer_nn_grid.jl
julia -t 32 --project=ext/IsingLearning ext/IsingLearning/experiments/MNIST/inlaid-55x55-pixel-readout/mnist_inlaid_input_training.jl
julia -t 32 --project=ext/IsingLearning ext/IsingLearning/experiments/MNIST/diagnostics/mnist_inlaid_input_diagnostics.jl
julia --project=ext/IsingLearning ext/IsingLearning/experiments/MNIST/plot_current_results.jl
julia --project=ext/IsingLearning ext/IsingLearning/experiments/plot_architecture_schematics.jl
```

The training files default to `32` `ProcessManager` workers. Start Julia with
`-t 32` for the intended run shape.
