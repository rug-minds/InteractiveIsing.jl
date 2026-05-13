# Normal LocalLangevin 2->4->1 XOR Grid

This run tunes only the ordinary EqProp route. Split-snapshot is intentionally disabled.

| Spec | Best MSE | Best Acc | T | stepsize | drift | free/nudged | β | lr | weight scale |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| `aggr_T005_s050_b10_lr0005` | 0.664998 | 0.75 | 0.005 | 0.5 | 1.0 | 900/900 | 1.0 | 0.0005 | 0.25 |
| `aggr_T010_s050_b10_lr0005` | 0.661457 | 1.0 | 0.01 | 0.5 | 1.0 | 900/900 | 1.0 | 0.0005 | 0.25 |
| `aggr_T020_s080_b10_lr0004` | 0.848513 | 0.75 | 0.02 | 0.8 | 1.0 | 1000/1000 | 1.0 | 0.0004 | 0.3 |
| `aggr_T010_s100_b15_lr0003` | 0.978129 | 0.75 | 0.01 | 1.0 | 1.0 | 1200/1200 | 1.5 | 0.0003 | 0.3 |

CSV: `normal_grid_metrics.csv`
Plot: `normal_grid_progress.png`
