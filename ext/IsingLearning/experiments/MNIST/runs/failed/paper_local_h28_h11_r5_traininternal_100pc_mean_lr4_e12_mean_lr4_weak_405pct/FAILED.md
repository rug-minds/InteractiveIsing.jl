# Failed Run: LR4 Mean-Normalized Paper-Local MNIST

Use of this file: explains why this run was kept under `failed` instead of `current`.

This run used the clean ProcessManager paper-local recipe with the old visible architecture (`784 -> 784 -> 121 -> 40`), radius 5, 32 workers, 100 train examples per class, 20 test examples per class, 50/50 free/nudged sweeps, and `gradient_normalization = :mean` with learning rates multiplied by four.

It improved slightly over the plain mean-normalized run, reaching best test accuracy 0.405 at epoch 11, but it is still not a good architecture result. It is kept because it shows that simply scaling the mean-normalized minibatch learning rate is not enough to recover the earlier online-per-sample behavior.
