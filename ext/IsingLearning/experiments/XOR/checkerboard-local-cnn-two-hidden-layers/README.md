# Checkerboard Local CNN-Like XOR

This folder is for checkerboard XOR experiments that use a local, CNN-like
spatial Ising graph. The input bits are not scalar spins: each truth-table case
is embedded as an `8x8` checkerboard field, so the graph has to move spatial
evidence through local hidden layers before producing the XOR class.

The active experiment is one architecture:

```text
8x8 checkerboard input -> 8x8 hidden1 -> 4x4 hidden2 -> 4x4 two-class output
```

The architecture is fixed while the local connectivity is swept:

- `r1`: local radius for input -> hidden1 and hidden1 recurrent couplings.
- `r2`: local radius for hidden1 -> hidden2, hidden2 recurrent couplings, and
  hidden2 -> output readout couplings.

The current grid is `r1 = 1:5` and `r2 = 1:2`. Those ten combinations are one
experiment because they test the same architecture, not ten different model
families.

## Motivation

Earlier checkerboard runs mixed hidden sizes, restart histories, and radius
ranges. That made the folder hard to read and made the results hard to compare.
This reset keeps one architecture fixed and asks a cleaner question: which
locality pairing lets the `8x8 -> 4x4` checkerboard architecture learn XOR?

The `8x8 -> 4x4` reduction is intentionally smaller than the older equal-size
hidden-layer runs. It is a first test of whether the CNN-like compression can
learn before trying larger or deeper variants.

## Output Code

The output layer is `4x4` and uses `two_class` targets. Half of the output
sites represent XOR false and half represent XOR true. The class score is the
mean of the matching output region, so the important metrics are:

- `mse`: score MSE over the four XOR truth-table cases.
- `accuracy`: whether the two-class decision is correct for each case.
- `min_margin`: the weakest signed class margin across the four cases.

Positive `min_margin` on all four cases matters more than a transient accuracy
hit, because it is more likely to survive repeated stochastic validation.

## Training

`xor_local_cnn_like_grid.jl` runs the grid through the process manager. Each
worker executes reusable contrastive process algorithms for free and nudged
phases, accumulates worker-local gradients, and the manager applies Adam.

Default settings for this experiment are:

- 32 workers.
- 100 epochs.
- 64 repeats per XOR case.
- 20 free sweeps and 20 nudged sweeps.
- Adam with `lr = 0.002`, `lr_decay = 0.995`, `lr_min = 0.0002`.
- coupling weight decay `1e-4`.
- zero initialization before each trajectory.

Run it with:

```powershell
julia -t 32 --project=ext/IsingLearning ext/IsingLearning/experiments/XOR/checkerboard-local-cnn-two-hidden-layers/xor_local_cnn_like_grid.jl
```

## Files

- `schematic.png`: architecture schematic for the active `8x8 -> 4x4`
  checkerboard experiment.
- `xor_local_cnn_like_grid.jl`: training and plotting script for this grid.
- `validate_checkerboard_checkpoints.jl`: high-repeat checkpoint validation.
- `experiments/current/input_8x8_hidden1_8x8_hidden2_4x4_output_4x4_two_class`:
  current result folder for this architecture. Each `r1_*_r2_*` subfolder holds
  metrics, checkpoints, and a learning plot for that radius pairing.
