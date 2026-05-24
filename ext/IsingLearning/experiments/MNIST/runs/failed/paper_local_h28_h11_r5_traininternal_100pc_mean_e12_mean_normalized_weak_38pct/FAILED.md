# Failed Run: Mean-Normalized Paper-Local MNIST

Use of this file: explains why this run was kept under `failed` instead of `current`.

This run used the clean ProcessManager paper-local recipe with the old visible architecture (`784 -> 784 -> 121 -> 40`), radius 5, 32 workers, 100 train examples per class, 20 test examples per class, 50/50 free/nudged sweeps, and `gradient_normalization = :mean`.

It did not collapse, but it only reached best test accuracy 0.38 after 12 epochs. It is kept because it shows that mean-normalizing the minibatch avoids the class-0 collapse seen with `:sum`, but learns much more weakly than the older online-per-sample implementation.
