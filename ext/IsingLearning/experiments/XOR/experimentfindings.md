# XOR Experiment Findings

Use of this file: essential findings needed to make XOR experiments run correctly. Keep this short; do not archive every failed run here.

- The current small direct baseline is the `2 -> 2x2 -> 4` majority-vote ProcessManager/Adam architecture under `two-input-2x2-hidden-majority-vote-baseline`.
- Parallelism requires enough manager jobs. With 4 XOR cases and 32 workers, use 8 chunks per case for 32 jobs per epoch.
- Do not make each random initialization its own manager job. Put repeats inside the worker `ProcessAlgorithm`; e.g. 32 repeats per case with 32 jobs means each job runs 4 repeats internally.
- For a less noisy gradient, increase repeats per case while keeping 32 jobs. Examples: 128 repeats gives 16 repeats per job; 256 repeats gives 32 repeats per job.
- Worker graphs should share static model data such as adjacency and base bias arrays. Worker state, target, and clamp buffers must stay worker-local.
- The active local checkerboard experiment is `8x8` input -> `8x8` hidden1 -> `4x4` hidden2 -> `4x4` two-class output, with separate `r1` and `r2` locality sweeps.
- Fixed SGD was brittle for these noisy contrastive gradients. The current XOR experiment files use `Optimisers.jl`; Adam is the default optimizer.
- A useful XOR result must show all four cases, per-case scores, margins, and stable correctness. A best-MSE bar plot without stable learning curves is not enough.
- Per-run PNGs belong beside their source CSVs under `experiments/current`. Architecture-level aggregate plots should live in the active architecture experiment folder.
- For local checkerboard experiments, rank configurations primarily by worst-case margin and stable all-correct epochs, not only by best MSE.
- In the `8x8 -> 4x4` checkerboard grid, `r2 = 2` is the important split. The best validated checkpoints were `r1_1_r2_2` and `r1_4_r2_2`; both reached sub-0.05 training MSE by epoch 100.
- The `r1_1_r2_2` best-margin checkpoint validated at 1024 repeats with MSE `0.0223`, accuracy `1.0`, and worst-case margin `0.7812`. Treat it as the first checkpoint to reuse for this architecture.
- The `r1_4_r2_2` best-margin checkpoint validated at 1024 repeats with MSE `0.0417`, accuracy `1.0`, and worst-case margin `0.7194`. It is the second useful checkpoint.
- The `r2 = 1` checkerboard configurations are not robust for this compressed hidden2 architecture. They can hit accuracy, but MSE stays high and margins are small or unstable.
- A 300-epoch sanity run of `xor_manager_input_averaging.jl` with its current direct defaults bounced between 0.25 and 0.75 accuracy. Do not use it as proof that the manager recipe is solved without retuning.
- The clean edge-input `16x16` sweep with zero-start `BlockLangevin`, replicated two-class output, 20/20 sweeps, Adam `lr=0.002`, and 64 repeats/case solved for wider local neighborhoods. In `edge_twoclass_zero_side16_nn1to10_e160`, NN `6`, `7`, and `9` had 512-repeat reload accuracy `1.0` with worst-case margins `0.6909`, `0.8680`, and `0.7884` respectively.
- In that same edge-input sweep, NN `1-4` did not solve by epoch 160. Treat too-local edge propagation as structurally insufficient for the current line-to-line XOR task.
