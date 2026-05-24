# 2 -> 8x8 -> 1 Split-Input High-Sweep Run

This run tested the cleaned split-input edge-signal XOR setup:

- two scalar input spins;
- input spin 1 connects to the upper half of the hidden layer left edge;
- input spin 2 connects to the lower half of the hidden layer left edge;
- the hidden layer is `8x8`;
- the output is one scalar spin connected from the hidden layer right edge;
- no input kick and no seeded hidden behavior.

The active runner was:

```julia
ext/IsingLearning/experiments/edge_signal_xor/run_split_input_high_sweeps.jl
```

The runner uses:

```julia
include("edge_signal_split_input_core.jl")
```

and then calls `train_split_input_regularized_xor`.

## Completed Setting

`sweeps160_160_val320`

- active spins: `65`
- free steps: `10400`
- nudged steps: `10400`
- validation steps: `20800`
- `minit = 3`
- `eval_repeats = 12`
- `beta = 0.20`
- `weight_decay = 0.003`
- nudged temperature factor: `1.10`
- learning rate: `0.00028`

Final logged result at epoch `6000`:

| epoch | MSE | accuracy | output means |
|---:|---:|---:|---|
| 6000 | 1.210246 | 0.50 | `[0.211, 0.096, -0.272, -0.031]` |

Best logged epoch by MSE in this completed setting:

| epoch | MSE | accuracy | output means |
|---:|---:|---:|---|
| 3000 | 0.868851 | 0.75 | `[-0.035, 0.332, 0.127, 0.156]` |

Saved artifacts:

- `sweeps160_160_val320/learning_metrics.csv`
- `sweeps160_160_val320/learning_progress.png`
- `sweeps160_160_val320/initial_graph.jld2`
- `sweeps160_160_val320/best_graph.jld2`

## Interrupted Setting

`sweeps240_240_val480`

This setting was stopped at epoch `4000` because it was clearly worse than the 160-sweep setting.
It did not reach the file-saving block, so no per-setting CSV/PNG was produced.

Observed terminal metrics:

| epoch | MSE | accuracy | output means |
|---:|---:|---:|---|
| 0 | 1.073611 | 0.75 | `[-0.212, -0.386, 0.086, -0.042]` |
| 1 | 1.111765 | 0.50 | `[-0.071, -0.329, -0.141, -0.282]` |
| 1000 | 1.469835 | 0.25 | `[-0.068, -0.524, -0.184, 0.135]` |
| 2000 | 1.453502 | 0.25 | `[-0.057, -0.588, -0.136, 0.054]` |
| 3000 | 1.544760 | 0.00 | `[0.063, -0.510, -0.188, 0.165]` |
| 4000 | 1.138245 | 0.25 | `[0.135, 0.284, -0.268, 0.070]` |

## Takeaway

Increasing the relaxation from roughly tens of sweeps to `160-240` sweeps did not fix the edge-signal task by itself. In this setup the outputs stayed small and noisy instead of settling into a reliable scalar XOR readout. More sweeps alone is therefore not the missing ingredient for `2 -> 8x8 -> 1`.
