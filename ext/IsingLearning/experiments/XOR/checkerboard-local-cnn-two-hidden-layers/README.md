# Checkerboard Local CNN-Like XOR

This experiment tests whether a local, spatial Ising architecture can learn XOR
when the two input bits are embedded as checkerboard patterns instead of two
scalar spins.

The architecture is:

```text
8x8 checkerboard input -> HxH hidden layer -> HxH hidden layer -> 4x4 output
```

Both hidden layers use local inter-layer connectivity. The main sweep changes
the hidden side length and the local connection radius. The retained current
runs use `H = 8` and non-periodic hidden layers.

## Motivation

The earlier scalar XOR experiments show whether a small graph can learn a clean
truth table. This checkerboard experiment is a harder spatial version: each XOR
case is presented as an image-like field, then the model has to move information
through two local hidden layers before producing a two-class output code.

This makes it useful for debugging MNIST-like local architectures without paying
the cost and ambiguity of the full MNIST task.

## Output And Metrics

The retained runs use `two_class` output mode. The `4x4` output layer is split
into two replicated class regions. XOR false targets class `0`; XOR true targets
class `1`.

The main metric is score MSE over the four XOR cases. Accuracy and margin are
logged as separate diagnostics:

- `accuracy`: whether the two-class decision is correct.
- `min_margin`: the weakest signed class margin across the four cases.
- `best_min_margin`: the best worst-case margin seen during a run.

High accuracy by itself is not enough. The useful checkpoints are the ones with
positive margins on all four cases, because those are more stable when
validation is repeated.

## Training Setup

The script uses the process manager with reusable contrastive workers. Each
worker runs free and nudged phases, accumulates a local contrastive gradient, and
the manager applies an Adam update after each batch.

The important optimizer fields are:

- `lr`: the initial Adam learning rate.
- `lr_decay`: multiplicative decay applied to the learning rate at each update.
- `lr_min`: lower bound on the decayed learning rate.
- `weight_decay`: L2 penalty on trainable couplings before the optimizer step.

Folder names now spell out learning-rate decay explicitly. For example,
`lr_decay0p999` means `lr_decay = 0.999`, so update `k` uses approximately
`lr * 0.999^(k-1)` until `lr_min` is reached.

## Current Runs

`local_checkerboard_two_class_zero_init_h8_r8_r9_repeats64_sweeps20_epochs80`
is the original zero-initialized short run for radii 8 and 9. Radius 8 learned
well; radius 9 did not.

`local_checkerboard_two_class_h8_r7_to_r10_resume_best_margin_e200_lr0p001_lr_decay0p998`
continues from best-margin checkpoints and compares radii 7 through 10 for 200
epochs. It is the best retained sweep overall; radius 10 reaches the lowest MSE
in this folder.

`local_checkerboard_two_class_h8_r9_r10_second_resume_best_margin_e240_lr0p0005_lr_decay0p999`
is a second continuation for radii 9 and 10 with a smaller initial learning rate
and gentler learning-rate decay. It checks whether the larger-radius runs become
more stable with slower updates.

## Files

- `xor_local_cnn_like_grid.jl`: experiment script. It no longer writes README
  summaries; it writes metrics, checkpoints, and plots only.
- `validate_checkerboard_checkpoints.jl`: high-repeat validation for saved
  checkpoints.
- `schematic.png`: architecture schematic.
- `experiments/current`: retained run folders.
- `aggregate_plots`: comparison plots collected from current runs.
