# Normal LocalLangevin 2->4->1 XOR Grid

This run tunes only the ordinary EqProp route. Split-snapshot is intentionally disabled.

| Spec | Best MSE | Best Acc | T | stepsize | drift | free/nudged | β | lr | weight scale |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| `T015_s015_b02_lr001_long` | 0.922177 | 0.75 | 0.015 | 0.15 | 0.5 | 500/500 | 0.2 | 0.001 | 0.22 |
| `T020_s020_b02_lr001_long` | 0.910481 | 0.75 | 0.02 | 0.2 | 0.6 | 500/500 | 0.2 | 0.001 | 0.22 |
| `T030_s025_b03_lr001_long` | 0.866022 | 0.75 | 0.03 | 0.25 | 0.7 | 500/500 | 0.3 | 0.001 | 0.22 |
| `T020_s030_b05_lr0007_long` | 0.872517 | 0.75 | 0.02 | 0.3 | 0.8 | 600/600 | 0.5 | 0.0007 | 0.25 |

CSV: `normal_grid_metrics.csv`
Plot: `normal_grid_progress.png`
