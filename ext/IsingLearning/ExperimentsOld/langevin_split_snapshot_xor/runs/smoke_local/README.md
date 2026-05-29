# Langevin Split-Snapshot XOR Search

This run tests a radical Langevin-only EqProp variant. The free phase captures an early state and a later free endpoint. Plus/minus nudged phases restart from the early state, but the contrastive gradient is computed with the free endpoint graph.

All configs use explicit bipolar frozen input, no input internal weights, output pattern clamping, symmetric adjacency, and no polynomial/double-well local potential.

| Config | Best MSE | Best Acc | Saved | Notes |
|---|---:|---:|---|---|
| `simple_local_s020_d035_T020_b10` | 1.356347 | 0.75 | no | split-snapshot Langevin: mode=local_langevin, side=2, hidden=4, stepsize=0.2, max_drift=0.35, Tfactor=0.02, β=1.0, free=500, early=40, nudged=700, nudged_T_bump=3.0 |

Metrics CSV: `langevin_split_snapshot_metrics.csv`
Progress PNG: `langevin_split_snapshot_progress.png`
