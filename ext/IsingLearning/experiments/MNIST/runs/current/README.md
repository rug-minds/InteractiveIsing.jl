# Current MNIST Runs

Use of this folder: curated MNIST runs that showed useful learning with the clean experiment files. Weak, collapsed, timed-out, or superseded runs are under `../failed`.

## CNN-Style Paper Manager

- `cnn_two_layer_h28_h14_r5_mean_lr8_b32_e30`
  - Architecture: `28^2 input fields -> 28^2 hidden1 -> 14^2 hidden2 -> 40 outputs`.
  - Recipe: ProcessManager, 32 workers, 32-sample minibatches, shared graph data, paper-style output-field nudge, `gradient_normalization = :mean`, 8x LR scale, trainable same-layer couplings.
  - Training slice: `100/class`; small test slice: `20/class`.
  - Best small-test accuracy: `0.80` at epoch 27.
  - Larger evals of best checkpoint:
    - `100/class`, 3 reads/50 sweeps: `0.791`.
    - `100/class`, 10 reads/75 sweeps: `0.854`.
    - `100/class`, 30 reads/75 sweeps: `0.861`.

- `paper_local_h28_h11_r5_traininternal_100pc_mean_lr16_b32_e30`
  - Architecture: `28^2 input fields -> 28^2 hidden1 -> 11^2 hidden2 -> 40 outputs`.
  - Recipe: ProcessManager, 32 workers, 32-sample minibatches, shared graph data, paper-style output-field nudge, `gradient_normalization = :mean`, 16x LR scale, trainable same-layer couplings.
  - Training slice: `100/class`; small test slice: `20/class`.
  - Best small-test accuracy: `0.755` at epoch 23.
  - Larger evals of best checkpoint:
    - `100/class`, 3 reads/50 sweeps: `0.719`.
    - `100/class`, 10 reads/75 sweeps: `0.801`.

- `paper_local_h28_h11_r7_traininternal_100pc_mean_lr16_b32_e20`
  - Same recipe but local fanout radius `7`.
  - Best small-test accuracy: `0.69` at epoch 18.
  - Kept as an architecture comparison; radius `5` is currently better.
