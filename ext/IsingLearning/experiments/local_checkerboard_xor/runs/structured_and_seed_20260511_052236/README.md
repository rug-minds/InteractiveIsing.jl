# Structured AND-Seed Checkerboard Search

This run tests whether the local checkerboard task needs an explicit local `A`, `B`, and `AND` feature route. The graph still uses symmetric weights, same-layer local connections, and the normal contrastive-gradient worker path.

| Rank | Config | Best MSE | Best Acc | Saved |
|---:|---|---:|---:|---|
| 1 | `structured_2x2_T0.001_ib2.0_f3.5_o0.8_ob0.4_b0.1_lr0.002` | 0.095703 | 1.0 | yes |

Metrics CSV: `structured_and_seed_metrics.csv`
Summary CSV: `structured_and_seed_summary.csv`
Progress PNG: `structured_and_seed_progress.png`
