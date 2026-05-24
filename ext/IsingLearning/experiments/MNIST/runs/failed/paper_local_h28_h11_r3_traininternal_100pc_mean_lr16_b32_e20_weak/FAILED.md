# Failed Run: Radius 3 Local Paper MNIST

Use of this file: explains why this radius-3 architecture run is under `failed`.

This used the same clean manager recipe as the successful radius-5 run: `28x28 -> 11x11 -> 40`, 32 workers, 32-sample minibatches, `gradient_normalization = :mean`, 16x LR scale, trainable same-layer couplings, `100/class` train, and `20/class` test.

Radius 3 did learn weakly but only reached best small-test accuracy `0.385` in 20 epochs. It is kept because it is a useful negative architecture comparison against radius 5 and radius 7.
