# Normal LocalLangevin 2->4->1 XOR Grid

This run tunes only the ordinary EqProp route. Split-snapshot is intentionally disabled.

| Spec | Best MSE | Best Acc | T | stepsize | drift | free/nudged | β | lr | weight scale |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| `cold_T001_s080_b10_lr0004_w035` | 1.254435 | 0.75 | 0.001 | 0.8 | 1.0 | 2500/2500 | 1.0 | 0.0004 | 0.35 |
| `cold_T002_s080_b10_lr0004_w035` | 0.628804 | 0.75 | 0.002 | 0.8 | 1.0 | 2500/2500 | 1.0 | 0.0004 | 0.35 |
| `cold_T005_s100_b10_lr0003_w04` | 1.017968 | 0.75 | 0.005 | 1.0 | 1.0 | 3000/3000 | 1.0 | 0.0003 | 0.4 |

CSV: `normal_grid_metrics.csv`
Plot: `normal_grid_progress.png`
