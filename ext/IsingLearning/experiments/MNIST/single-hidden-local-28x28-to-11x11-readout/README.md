# Single-Hidden Local MNIST Readout

Architecture: `28x28` MNIST image fields drive an `11x11` local hidden layer,
then a dense readout produces `40` output spins, four replicas per digit.

- `schematic.png`: generated architecture schematic.
- `mnist_local_manager_grid.jl`: manager-backed training script.
- `evaluate_manager_checkpoint.jl`: post-hoc checkpoint evaluator.
- `experiments/current`: retained successful training/evaluation runs.
- `aggregate_plots`: comparison plots for this architecture family.
