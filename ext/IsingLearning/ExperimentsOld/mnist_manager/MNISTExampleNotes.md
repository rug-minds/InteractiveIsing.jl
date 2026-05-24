# MNIST Learning Examples

This note tracks the example scripts in `examples/Learning` that use the
current shared-data `ProcessManager` MNIST path.

## Shared Manager MNIST

File:

- `examples/Learning/MNISTTraining.jl`

Architecture:

- `784 -> 120 -> 40`
- The output layer is ten classes with four spins per class.

Runtime setup:

- Defaults to `32` workers.
- Uses `LocalLangevin` with cyclic order.
- Uses `init_mnist_trainer(...; share_static_model_data = true, input_mode = :field)`.
- Worker graphs share the source graph adjacency and learnable base bias.
- Each worker has its own input-pattern field, state, clamping storage, and
  gradient buffer.

Default output folder:

- `ext/IsingLearning/experiments/mnist_manager/runs/<timestamp>_example_mnist`

Main environment controls:

- `ISING_MNIST_WORKERS`
- `ISING_MNIST_EPOCHS`
- `ISING_MNIST_BATCHSIZE`
- `ISING_MNIST_SWEEPS`
- `ISING_MNIST_TRAIN_LIMIT`
- `ISING_MNIST_VALIDATION_LIMIT`
- `ISING_MNIST_OUTDIR`

The script writes:

- `mnist_shared_training_summary.csv`
- `mnist_shared_training.bin`

Smoke/full-run checks:

- Smoke run:
  - output folder: `runs/20260522_example_mnist_smoke`
  - workers `2`, batch `4`, train limit `8`, sweeps `1`.
  - completed and wrote a checkpoint.
- Full 32-worker epoch run:
  - output folder: `runs/20260522_example_mnist_32worker_epoch`
  - workers `32`, batch `256`, full `60_000` train samples, no validation,
    sweeps `500`.
  - elapsed `241.81s`, about `4.03 min`.

Important current caveat:

- This stock manager path uses symmetric plus/minus equilibrium propagation
  with direct quadratic clamping on all output spins. After fixing field-input
  bugs it reduces output MSE, but it has not produced reliable MNIST
  classification accuracy yet.
- Two source bugs were fixed during the 2026-05-22 pass:
  - precomputed field inputs now also disable the input layer, matching
    `apply_input`;
  - field-input training adds the missing input-to-hidden gradient terms,
    because the input state is intentionally cleared in field mode.
- Fresh diagnostics:
  - `runs/20260522_mnist_gradient_fix_4096x10`: before disabling the input
    layer in the pattern path, MSE improved but accuracy stayed random.
  - `runs/20260522_mnist_field_input_fix_2048x8`: after both fixes, MSE
    improved further (`~24.4 -> ~21.4` on the train probe), but accuracy still
    stayed near random.
  - `runs/20260522_balanced_shared_target_sweep`,
    `runs/20260522_balanced_shared_sign_negative`, and
    `runs/20260522_balanced_stronger_weights` show that off-target coding,
    update sign, and larger initial couplings did not by themselves make this
    quadratic-clamp path classify.

## Working Paper-Style MNIST

File:

- `ext/IsingLearning/experiments/mnist_manager/mnist_paper_like_ep.jl`

Architecture:

- `784 -> 120 -> 40`, but the input layer is represented as an external
  input-to-hidden field map, matching the Laydevant et al. setup.
- The sampled Ising graph is hidden/output only: `120 -> 40`.
- The 40 output spins are four replicas per digit.

Learning rule:

- One-sided paper-style EP: free sample, then nudged sample from the free state.
- Nudging is an output-field shift by the target, not quadratic output
  clamping.
- Couplings and fields are negated when installed in this package's graph,
  because the package Hamiltonian sign is opposite to the paper's `dimod`
  convention.

Fresh reproduction:

- `runs/20260522_paper_like_repro_100pc_r3_s50_e8`
- Settings: train `100/class`, test `10/class`, reads `3/3`, sweeps `50/50`,
  epochs `8`, hidden `120`, output replicas `4`.
- Best balanced test accuracy: `83%` at epoch `6`.
- Checkpoints:
  - `best_model.bin`
  - `final_model.bin`
- Stronger fresh run:
  - `runs/20260522_paper_like_repro_200pc_r10_s100_e15`
  - Settings: train `200/class`, test `100/class`, reads `10/10`,
    sweeps `100/100`, epochs `15`.
  - Best balanced test accuracy: `85.0%` at epoch `8`.
  - Epoch time after the first epoch was about `18-20s`; first epoch was
    about `29s`.

Supporting diagnostics:

- `learning_sweeps/MNISTRawPixelRidgeCheck.jl` confirms the balanced data and
  label path: raw-pixel ridge reached `65%` accuracy on `100/class` train and
  `30/class` validation.
- The hidden-state ridge warm-start file was updated to use the current
  shared field-input path and to write hidden-output readout weights
  symmetrically. It still showed weak random-feature accuracy, which supports
  using the paper-style field/nudge formulation rather than the stock direct
  clamp for now.

## Local Paper-Style MNIST

File:

- `ext/IsingLearning/experiments/mnist_manager/learning_sweeps/MNISTLocalPaperLikeEP.jl`

Learning schema:

- Same one-sided paper-style EP as `mnist_paper_like_ep.jl`.
- Input pixels are external fields into the first hidden layer.
- The sampled graph is hidden1/hidden2/output.
- Trainable matrices:
  - local input-to-hidden1 field map;
  - local hidden1-to-hidden2 couplings;
  - dense hidden2-to-output couplings;
  - hidden/output biases.
- The sampled graph can also include fixed local intra-layer couplings on both
  hidden layers and the 40-spin output layer.

Runs:

- Smoke:
  - `runs/20260522_local_paper_smoke`
  - `14x14 -> 14x14 -> 40`, `2/class`, one epoch, short sweeps.
  - completed and wrote checkpoints.
- First signal:
  - `runs/20260522_local_paper_h14_r4_s50_lr003_nointernal`
  - `14x14 -> 14x14 -> 40`, `20/class`, no fixed intra-layer couplings.
  - best small-test accuracy `32%`.
- Better local shape:
  - `runs/20260522_local_paper_h28_h14_r5_s50_lr003_50pc`
  - `28x28 -> 14x14 -> 40`, `50/class`, no fixed intra-layer couplings.
  - best `10/class` test accuracy `74%`.
- Best current local run:
  - `runs/20260522_local_paper_h28_h11_r5_s50_lr003_100pc`
  - `28x28 -> 11x11 -> 40`, `100/class`, no fixed intra-layer couplings.
  - best `20/class` test accuracy `75.5%`.
  - checkpoint eval in `runs/20260522_local_paper_h28_h11_best_eval100pc`:
    `70.6%` on `100/class` test.
- Trainable intra-layer couplings:
  - `MNISTLocalPaperLikeEP.jl` now supports trainable local same-layer
    couplings with `ISING_MNIST_LOCAL_PAPER_TRAIN_INTERNAL=true`.
  - The fixed intra-layer couplings remain available when that flag is false.
  - Best overnight local result so far:
    `runs/20260522_local_paper_h28_h11_traininternal_200pc`
  - Settings: `28x28 -> 11x11 -> 40`, local radius `5`, trainable
    same-layer radius `1`, `200/class` train, `50/class` test,
    reads `3/3`, sweeps `50/50`, learning rates
    `W0/W12/W2O=0.003`, `W11/W22/WOO=0.0005`.
  - Best in-run `50/class` test accuracy: `75.8%` at epoch `5`.
  - Larger eval:
    `runs/20260522_local_paper_h28_h11_traininternal_200pc_best_eval100pc`
    reached `76.3%` on `100/class` test.
  - Smaller `100/class` train run
    `runs/20260522_local_paper_h28_h11_traininternal_100pc` reached `77.5%`
    on a `20/class` test slice and `73.9%` on `100/class`.
  - Output-only same-layer training reached `76.5%` on the `20/class` test
    slice, so hidden same-layer updates appear mildly useful but not dominant.
  - Negative checks:
    - `runs/20260522_local_paper_h28_h11_traininternal_100pc_s100` used
      `100/100` sweeps and peaked at `72.5%`, so the current local bottleneck
      is not simply too few Metropolis sweeps.
    - `runs/20260522_local_paper_h28_h11_traininternal_r7_100pc` widened the
      local fanout to radius `7` and peaked at `74%`.
    - `runs/20260522_local_paper_h28_h11_traininternal_halfinternal_100pc`
      halved same-layer learning rates to `0.00025` and peaked at `68.5%`.
  - Working continuation schedule:
    - start from
      `runs/20260522_local_paper_h28_h11_traininternal_200pc/best_model.bin`;
    - continue with `W0/W12/W2O lr = 0.001` and
      `W11/W22/WOO lr = 0.00015`;
    - run folder:
      `runs/20260522_local_paper_h28_h11_200pc_best_continue_lr001`;
    - best `50/class` test accuracy: `83.0%`;
    - large eval folder:
      `runs/20260522_local_paper_h28_h11_continue_lr001_best_eval100pc`;
    - `3` free reads: `80.2%` on `100/class`.
  - Second continuation:
    - start from the previous continued best checkpoint;
    - continue with `W0/W12/W2O lr = 0.0005` and
      `W11/W22/WOO lr = 0.000075`;
    - run folder: `runs/20260522_local_paper_h28_h11_continue_lr0005`;
    - best `50/class` test accuracy: `82.8%`;
    - `100/class` eval:
      - `3` free reads: `80.7%`;
      - `10` free reads: `87.4%`;
      - `30` free reads: `87.9%`.
  - Inference grid on the second-continuation checkpoint:
    - `mean_reads`, `10` reads, `50` sweeps: `79.8%`;
    - `mean_reads`, `30` reads, `50` sweeps: `84.2%`;
    - `mean_reads`, `10` reads, `75` sweeps: `82.6%`;
    - `best_energy`, `10` reads, `75` sweeps: `86.6%`;
    - `best_energy`, `30` reads, `75` sweeps: `88.2%`.
  - Best current continuation:
    - start from
      `runs/20260522_local_paper_h28_h11_continue_lr0005/best_model.bin`;
    - continue with `5/5` training reads, `W0/W12/W2O lr = 0.0005`,
      `W11/W22/WOO lr = 0.000075`;
    - run folder: `runs/20260522_local_paper_h28_h11_continue_lr0005_reads5`;
    - best `50/class` test accuracy: `89.0%`;
    - eval folder:
      `runs/20260522_local_paper_h28_h11_reads5_best_eval100pc_r30_s75`;
    - `100/class`, `30` reads, `75` sweeps: `91.0%`;
    - larger eval folder:
      `runs/20260522_local_paper_h28_h11_reads5_best_eval200pc_r30_s75`;
    - `200/class`, `30` reads, `75` sweeps: `89.0%`.
  - A lower `0.00025` continuation from the same checkpoint did not improve
    the `50/class` validation slice; it peaked at `87.6%`.
  - More-data continuation:
    - from the 5-read checkpoint, train on `500/class` with
      `W0/W12/W2O lr = 0.0002`, `W11/W22/WOO lr = 0.00003`;
    - run folder:
      `runs/20260522_local_paper_h28_h11_500pc_continue_lr0002_reads5`;
    - best `50/class` test accuracy: `87.8%`;
    - larger eval with `50` reads and `75` sweeps:
      `runs/20260522_local_paper_h28_h11_500pc_best_eval200pc_r50_s75`;
    - `200/class` accuracy: `89.7%`.
  - Current best:
    - from the 500/class checkpoint, train on `1000/class` with
      `W0/W12/W2O lr = 0.0001`, `W11/W22/WOO lr = 0.000015`;
    - run folder:
      `runs/20260522_local_paper_h28_h11_1000pc_continue_lr0001_reads5`;
    - larger eval:
      `runs/20260522_local_paper_h28_h11_1000pc_best_eval200pc_r50_s75`;
    - `200/class`, `50` reads, `75` sweeps: `90.65%`.
  - Additional negative checks:
    - `TARGET_OFF=0` collapsed toward a few classes and peaked at `30%`;
    - `beta=10` peaked at `74.5%`;
    - `beta=2.5` peaked at `74.5%`;
    - `OUTPUT_REPLICAS=8` peaked at `74.5%`;
    - `OUTPUT_LAYOUT=replica_digit` for four replicas peaked at `74%`;
    - inter-layer radius `3` peaked at `68%`.

Current interpretation:

- The local paper-style schema learns, unlike the stock quadratic-clamp manager
  path.
- A full `28x28 -> 28x28 -> 40` second hidden layer collapsed more easily in
  the short runs.
- Fixed intra-layer couplings hurt the first useful `28x28 -> 14x14 -> 40`
  run. Trainable same-layer couplings are better than fixed random same-layer
  couplings and improve the larger-eval local result from `70.6%` to `76.3%`,
  and the continuation with more training/evaluation reads reaches `91.0%` on
  a balanced `100/class` test slice and `90.65%` on `200/class`.
- The main remaining practical issue is sampling variance: the same checkpoint
  is much better with `30` low-energy reads than with the quick `3`-read
  setting.

## Local CNN-Like MNIST

File:

- `examples/Learning/MNISTLocalCNNLikeTraining.jl`

Architecture:

- `28x28 input -> 28x28 hidden -> 28x28 hidden -> 40 output`
- The two spatial inter-layer connections use a local square neighborhood.
- The default local radius is `5`, so each spatial unit can connect to an
  `11x11` window in the next spatial layer.
- Hidden layer 1, hidden layer 2, and the output layer now also receive local
  intra-layer couplings. Input-layer intra couplings are intentionally left out
  because field-mode input disables the input layer during sampling.
- The final hidden-to-output readout is dense. The output is class-like rather
  than spatial, so keeping the readout dense avoids only connecting the
  top-left part of the hidden map to the output layer. The 40-spin output layer
  still has local intra-layer couplings on top of that dense readout.

Runtime setup:

- Defaults to `32` workers.
- Uses the same shared-data trainer path as the paper-like example.
- The source edit needed for this example is small: shared worker construction
  now preserves the prototype graph layer layout instead of assuming one hidden
  layer.

Default output folder:

- `ext/IsingLearning/experiments/mnist_manager/runs/<timestamp>_example_mnist_local`

Main environment controls:

- `ISING_MNIST_LOCAL_WORKERS`
- `ISING_MNIST_LOCAL_EPOCHS`
- `ISING_MNIST_LOCAL_BATCHSIZE`
- `ISING_MNIST_LOCAL_SWEEPS`
- `ISING_MNIST_LOCAL_RADIUS`
- `ISING_MNIST_LOCAL_HIDDEN_INTERNAL_RADIUS`
- `ISING_MNIST_LOCAL_OUTPUT_INTERNAL_RADIUS`
- `ISING_MNIST_LOCAL_HIDDEN_INTERNAL_WEIGHT_SCALE`
- `ISING_MNIST_LOCAL_OUTPUT_INTERNAL_WEIGHT_SCALE`
- `ISING_MNIST_LOCAL_TRAIN_LIMIT`
- `ISING_MNIST_LOCAL_VALIDATION_LIMIT`
- `ISING_MNIST_LOCAL_OUTDIR`

The script writes:

- `mnist_local_cnn_like_training_summary.csv`
- `mnist_local_cnn_like_training.bin`

Smoke check:

- Output folders: `runs/20260522_example_mnist_local_smoke` and
  `runs/20260522_example_mnist_local_smoke_shape`
- workers `2`, batch `4`, train limit `8`, sweeps `1`, local radius `2`.
- completed and wrote a checkpoint.
- Intra-layer smoke:
  - output folder: `runs/20260522_local_cnn_intralayer_smoke`
  - workers `2`, batch `4`, train limit `8`, sweeps `1`, local radius `1`,
    hidden/output internal radius `1`.
  - completed and wrote a checkpoint.

## Folder Separation

The two examples write to separate timestamped run folders by default. Override
the specific `*_OUTDIR` variable when a run should be grouped under a named
experiment folder.
