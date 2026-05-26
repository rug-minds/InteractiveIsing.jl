# XOR Experiment Findings

Use of this file: essential findings needed to make XOR experiments run correctly. Keep this short; do not archive every failed run here.

- The current small direct baseline is the `2 -> 2x2 -> 4` majority-vote ProcessManager/Adam architecture under `two-input-2x2-hidden-majority-vote-baseline`.
- Parallelism requires enough manager jobs. With 4 XOR cases and 32 workers, use 8 chunks per case for 32 jobs per epoch.
- Do not make each random initialization its own manager job. Put repeats inside the worker `ProcessAlgorithm`; e.g. 32 repeats per case with 32 jobs means each job runs 4 repeats internally.
- For a less noisy gradient, increase repeats per case while keeping 32 jobs. Examples: 128 repeats gives 16 repeats per job; 256 repeats gives 32 repeats per job.
- Worker graphs should share static model data such as adjacency and base bias arrays. Worker state, target, and clamp buffers must stay worker-local.
- The local checkerboard graph stores only structural local edges for the checked radii; stored sparse entries matched nonzero entries for hidden sizes `8` and `16` at radii `1`, `5`, and `10`.
- Fixed SGD was brittle for these noisy contrastive gradients. The current XOR experiment files use `Optimisers.jl`; Adam is the default optimizer.
- A useful XOR result must show all four cases, per-case scores, margins, and stable correctness. A best-MSE bar plot without stable learning curves is not enough.
- Aggregate comparison PNGs belong in the family `aggregate_plots` folders; per-run PNGs belong beside their source CSVs under `experiments/current`. Use line style groups for NN/radius comparisons because color-only plots are too hard to read.
- The recent local CNN-like XOR grid with MSE around 0.85-1.2 and bouncing accuracy is not a useful result. Treat it as a failed diagnostic, not evidence of architecture ordering.
- For local checkerboard experiments, rank configurations primarily by worst-case margin and stable all-correct epochs, not only by best MSE.
- The first local checkerboard manager runs used majority-vote output with no weight decay and only `LocalLangevin`; this produced transient all-correct points with tiny margins, not stable learning.
- A later `BlockLangevin` run with hidden `8x8`, radii `8/9`, `β=1`, `T=0.001`, and 32 manager jobs also reached transient all-correct states, but margins stayed around `0.005-0.008` and drifted. Treat majority output as not yet stable.
- Pattern-output grid with 128 repeats and 64-repeat validation found logged best margins around `0.0456` for `h8/r2` and `h16/r5`, but 256-repeat reevaluation rejected those checkpoints. Treat them as promising but not solved.
- Zero initialization is a dead end for the pattern-output local checkerboard runs: scores stay near zero for both `8x8` and `16x16` hidden candidates.
- Stronger majority-output coupling with high-repeat validation also kept scores tiny and did not improve margins on the promising `r2/r5` candidates.
- Raising pattern-output `β` to `1.0` gave small all-correct points (`h8/r5` around `0.020`, `h8/r2` around `0.013`) but still no robust margin or retention.
- Direct replicated two-class output changes the local CNN-like result: with `zero` init, `h8/r8/open`, 20 free/nudged sweeps, `β=1`, Adam `lr=0.002`, and 64 repeats/case, the run reached 1.0 accuracy by epoch 25 and final worst-case margin `0.6358` at epoch 80.
- The `h8/r8/open` best-margin checkpoint from `local_checkerboard_twoclass_zero_h8_r8r9_rep64_s20_e80` passed a 512-repeat reload validation: score MSE `0.0490`, accuracy `1.0`, worst-case margin `0.6517`, scores `[-0.9909, 0.9264, 0.7368, -0.6517]`.
- In that same run, `h8/r9/open` improved but did not solve by epoch 80; it stayed at 0.75 accuracy with worst-case margin `-0.0356`.
- Longer two-class checkerboard training fixed the non-generalizing result. `local_checkerboard_twoclass_h8_r7to10_rep64_eval256_e240_lr002_decay995` solved h8 radii 7, 8, and 9 by 240 epochs, with r10 improving before the run was interrupted at epoch 90.
- Continuing from `best_margin_params.bin` with `lr=0.001`, `decay=0.998`, 200 more epochs, and 256 eval repeats gave solved h8 radii 7-10. Best logged margins: r7 `0.9555`, r8 `1.1239`, r9 `1.2208`, r10 `1.0790`.
- A second focused continuation of r9/r10 with `lr=0.0005`, `decay=0.999`, and 240 epochs improved r9 to logged margin `1.2329` and r10 to `1.1734`.
- 1024-repeat validation of the best focused two-class checkerboard checkpoints passed: r9 accuracy `1.0`, worst-case margin `1.2266`; r10 accuracy `1.0`, worst-case margin `1.1704`. The validated r9 checkpoint is the current best checkerboard XOR result.
- A 300-epoch sanity run of `xor_manager_input_averaging.jl` with its current direct defaults bounced between 0.25 and 0.75 accuracy. Do not use it as proof that the manager recipe is solved without retuning.
- The clean edge-input `16x16` sweep with zero-start `BlockLangevin`, replicated two-class output, 20/20 sweeps, Adam `lr=0.002`, and 64 repeats/case solved for wider local neighborhoods. In `edge_twoclass_zero_side16_nn1to10_e160`, NN `6`, `7`, and `9` had 512-repeat reload accuracy `1.0` with worst-case margins `0.6909`, `0.8680`, and `0.7884` respectively.
- In that same edge-input sweep, NN `1-4` did not solve by epoch 160. Treat too-local edge propagation as structurally insufficient for the current line-to-line XOR task.
