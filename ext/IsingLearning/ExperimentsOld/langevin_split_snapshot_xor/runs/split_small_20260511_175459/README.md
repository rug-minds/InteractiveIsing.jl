# Langevin Split-Snapshot XOR Search

This run tests a radical Langevin-only EqProp variant. The free phase captures an early state and a later free endpoint. Plus/minus nudged phases restart from the early state, but the contrastive gradient is computed with the free endpoint graph.

All configs use explicit bipolar frozen input, no input internal weights, output pattern clamping, symmetric adjacency, and no polynomial/double-well local potential.

| Config | Best MSE | Best Acc | Saved | Notes |
|---|---:|---:|---|---|
| `simple_block_s020_T020_b10` | 0.781766 | 1.0 | no | split-snapshot Langevin: mode=langevin, side=2, hidden=4, stepsize=0.2, Tfactor=0.02, β=1.0, free=350, early=30, nudged=500, nudged_T_bump=3.0 |
| `simple_global_s020_T020_b10` | 1.221393 | 0.75 | no | split-snapshot Langevin: mode=global_langevin, side=2, hidden=4, stepsize=0.2, Tfactor=0.02, β=1.0, free=350, early=30, nudged=500, nudged_T_bump=3.0 |
| `simple_global_s035_T020_b10` | 0.772355 | 0.75 | no | split-snapshot Langevin: mode=global_langevin, side=2, hidden=4, stepsize=0.35, Tfactor=0.02, β=1.0, free=350, early=20, nudged=600, nudged_T_bump=4.0 |

Metrics CSV: `langevin_split_snapshot_metrics.csv`
Progress PNG: `langevin_split_snapshot_progress.png`
