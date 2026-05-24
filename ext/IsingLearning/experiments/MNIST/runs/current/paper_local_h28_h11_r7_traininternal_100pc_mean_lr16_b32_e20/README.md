# Radius 7 Local Paper MNIST

Use of this folder: architecture comparison run for the clean ProcessManager local paper-style MNIST trainer.

This run used `28x28 -> 11x11 -> 40`, local fanout radius `7`, 32 workers, 32-sample minibatches, `gradient_normalization = :mean`, 16x LR scale, trainable same-layer couplings, `100/class` train, and `20/class` test.

It reached best small-test accuracy `0.69` at epoch 18. This is useful, but weaker than the radius-5 run, which reached `0.755` on the same small test slice and `0.801` on `100/class` with 10 reads/75 sweeps.
