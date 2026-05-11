# Structured AND-Seed Checkerboard Search

This run tests whether the local checkerboard task needs an explicit local `A`, `B`, and `AND` feature route. The graph still uses symmetric weights, same-layer local connections, and the normal contrastive-gradient worker path.

| Rank | Config | Best MSE | Best Acc | Saved |
|---:|---|---:|---:|---|
| 1 | `structured_2x2_T0.001_ib2.0_f3.5_o0.8_ob0.4_b0.1_lr0.002` | 0.060547 | 1.0 | yes |
| 2 | `structured_2x2_T0.001_ib2.0_f3.5_o0.7_ob0.6_b0.1_lr0.002` | 0.075195 | 1.0 | yes |
| 3 | `structured_2x2_T0.001_ib2.0_f3.5_o0.7_ob0.8_b0.2_lr0.002` | 0.075195 | 1.0 | yes |
| 4 | `structured_2x2_T0.001_ib2.0_f3.5_o0.8_ob0.4_b0.2_lr0.002` | 0.110352 | 1.0 | yes |
| 5 | `structured_2x2_T0.001_ib2.0_f3.5_o0.7_ob0.6_b0.2_lr0.004` | 0.12207 | 1.0 | yes |
| 6 | `structured_2x2_T0.001_ib2.0_f3.5_o0.7_ob0.8_b0.1_lr0.002` | 0.134766 | 1.0 | yes |
| 7 | `structured_2x2_T0.001_ib2.0_f3.5_o0.7_ob0.8_b0.2_lr0.004` | 0.134766 | 1.0 | yes |
| 8 | `structured_2x2_T0.001_ib2.0_f3.5_o0.7_ob0.4_b0.1_lr0.002` | 0.146484 | 1.0 | yes |
| 9 | `structured_2x2_T0.001_ib2.0_f3.5_o0.8_ob0.4_b0.2_lr0.004` | 0.154297 | 1.0 | yes |
| 10 | `structured_2x2_T0.001_ib2.0_f3.5_o0.7_ob0.4_b0.2_lr0.002` | 0.189453 | 1.0 | yes |
| 11 | `structured_2x2_T0.001_ib2.0_f3.5_o0.7_ob0.4_b0.2_lr0.004` | 0.195312 | 1.0 | yes |
| 12 | `structured_2x2_T0.001_ib2.0_f3.5_o0.7_ob0.6_b0.2_lr0.002` | 0.250977 | 1.0 | no |

Metrics CSV: `structured_and_seed_metrics.csv`
Summary CSV: `structured_and_seed_summary.csv`
Progress PNG: `structured_and_seed_progress.png`
