# Edge-Driven Single-Layer XOR Readout

This experiment applies each XOR input along one edge of a `16x16` Ising layer
and reads the opposite edge:

```text
16-spin checkerboard input edge -> 16x16 hidden field -> opposite 16-spin readout edge
```

The main script is `xor_edge_application_grid.jl`. The current useful recipe is
zero-start `BlockLangevin`, replicated two-class edge readout, 20 free/nudged
sweeps, Adam with learning-rate decay, explicit weight decay, and local
neighborhood sweeps up to NN 10.

- `experiments/current`: retained successful edge experiments, with per-run plots beside their CSV files and a copy of the simulation file.
- `aggregate_plots`: NN comparison plots for the edge architecture.
- `schematic.png`: edge-input and edge-readout architecture schematic.
