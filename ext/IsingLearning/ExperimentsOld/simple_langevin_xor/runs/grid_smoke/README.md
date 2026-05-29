# Normal LocalLangevin 2->4->1 XOR Grid

This run tunes only the ordinary EqProp route. Split-snapshot is intentionally disabled.

| Spec | Best MSE | Best Acc | T | stepsize | drift | free/nudged | β | lr | weight scale |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| `T010_s020_b02_lr003` | 1.180537 | 0.75 | 0.01 | 0.2 | 0.6 | 250/250 | 0.2 | 0.003 | 0.18 |

CSV: `normal_grid_metrics.csv`
Plot: `normal_grid_progress.png`
