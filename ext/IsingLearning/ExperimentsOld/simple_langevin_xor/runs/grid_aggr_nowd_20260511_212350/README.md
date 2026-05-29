# Normal LocalLangevin 2->4->1 XOR Grid

This run tunes only the ordinary EqProp route. Split-snapshot is intentionally disabled.

| Spec | Best MSE | Best Acc | T | stepsize | drift | free/nudged | β | lr | weight scale |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| `aggr_T010_s050_b10_lr0005` | 0.663922 | 0.75 | 0.01 | 0.5 | 1.0 | 900/900 | 1.0 | 0.0005 | 0.25 |

CSV: `normal_grid_metrics.csv`
Plot: `normal_grid_progress.png`
