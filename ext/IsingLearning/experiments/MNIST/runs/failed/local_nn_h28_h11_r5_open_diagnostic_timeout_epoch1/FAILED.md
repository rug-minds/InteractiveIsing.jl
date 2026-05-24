# Failed Run: Stock Local NN Diagnostic

Use of this file: explains why this stock `mnist_local_nn_grid.jl` run was stopped and kept under `failed`.

This tested the LocalLangevin/quadratic-clamp local NN path with `28x28 -> 28x28 -> 11x11 -> 40`, radius `5`, open hidden layers, 32 workers, batchsize `256`, `1024` training examples, and `256` validation examples.

It completed only one epoch before the 5-minute command timeout and was still near chance: train accuracy `0.082`, validation accuracy `0.070`. This is not a usable learning run. It is kept because it confirms that the stock local NN path is currently much slower and weaker than the paper-style manager recipe for this architecture.
