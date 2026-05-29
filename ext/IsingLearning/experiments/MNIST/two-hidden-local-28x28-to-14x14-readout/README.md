# Two-Hidden Local MNIST Readout

Architecture: `28x28` MNIST image fields drive a `28x28` local hidden layer,
then a `14x14` second hidden layer, then a dense `40`-spin readout.

- `schematic.png`: generated architecture schematic.
- `mnist_cnn_two_layer_nn_grid.jl`: manager-backed two-hidden-layer grid.
- `mnist_local_paper_manager_grid.jl`: shared manager recipe used by the grid.
- `experiments/current`: retained successful training/evaluation runs.
- `aggregate_plots`: comparison plots for this architecture family.

