# Edge-Driven Single-Layer XOR Readout

This experiment applies each XOR input through a separate `16x1` input layer,
propagates it through a full dynamic `16x16` Ising layer, and reads a separate
`16x1` output layer:

```text
16x1 input line -> 16x16 dynamic field -> 16x1 output line
```

The input line is not the first frozen column of the `16x16` field. The graph
has separate input and output line layers coupled to the left and right edges of
the dynamic field.

The main script is `xor_edge_application_grid.jl`. The current useful recipe is
zero-start `BlockLangevin`, replicated two-class edge readout, 20 free/nudged
sweeps, Adam with learning-rate decay, explicit weight decay, and local
neighborhood sweeps up to NN 10. The active robustness grid runs multiple seeds
per NN because single-seed results were too easy to overinterpret.

## Readout

The default readout is `two_class`, not a single majority vote over all 16
output spins. The final `16x1` output line is split into two replicated class
regions:

- sites `1:8`: XOR false class
- sites `9:16`: XOR true class

For each XOR case, validation averages the relaxed output line over
`eval_repeats`. The scalar decision score is then:

```text
score = mean(output[9:16]) - mean(output[1:8])
```

The prediction is XOR true when `score > 0` and XOR false otherwise. The logged
`mse` is the MSE between this scalar score and the target `-1` or `+1`.
`min_margin` is the weakest signed score across the four truth-table cases.

There is also a legacy `majority` mode in the script. In that mode the score is
`mean(output[1:16])`, so it really is a full-line majority-style readout. The
current experiments use `two_class` because it has been more stable.

- `experiments/current`: retained edge experiments, with per-seed plots beside their CSV files and a copy of the simulation file.
- `aggregate_plots`: NN comparison plots for the edge architecture.
- `schematic.png`: edge-input and edge-readout architecture schematic.
