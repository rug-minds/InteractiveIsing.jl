# Separate Input/Output Edge Lines, Side 16, NN 1-10, Seeds 1-5

This run tests robustness of the edge-driven XOR architecture across multiple
random seeds per local-neighborhood size.

The architecture is:

```text
16x1 input line -> 16x16 dynamic field -> 16x1 output line
```

The input and output lines are separate layers. They are not frozen columns
inside the `16x16` field. The input line is coupled to the left edge of the
dynamic field, and the output line is coupled to the right edge.

## Grid

- NN values: `1` through `10`.
- Seeds per NN: `1` through `5`.
- Output mode: replicated `two_class` output line.
- Dynamics: zero-start `BlockLangevin`.
- Sweeps: 20 free and 20 nudged.
- Optimizer: Adam with learning-rate decay and coupling weight decay.
- Worker setup: 32 manager workers, 64 repeats per XOR case.

Each `nn*_seed*` subfolder contains one seed run with `metrics.csv`,
`learning_curve.png`, and saved checkpoints.

Aggregate files:

- `metrics.csv`: all logged epochs for all NN/seed combinations.
- `summary.csv`: one best-result row per NN/seed combination.
- `learning_summary.png`: aggregate learning and best-margin plot.
- `robustness_summary.csv`: per-NN aggregate over seeds.
- `robustness_summary.png`: solved fraction, mean best MSE, and mean best
  margin by NN.
