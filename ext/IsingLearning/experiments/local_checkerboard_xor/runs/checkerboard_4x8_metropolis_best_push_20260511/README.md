# Checkerboard 4x8 Metropolis Best Push

Narrow follow-up around the first corrected checkerboard run that approached `0.1` MSE. All configs use fully frozen bipolar input, no input internal weights, pattern output clamping, no polynomial/double-well local potentials, and symmetric Metropolis dynamics.

| Config | Best MSE | Best Acc | Saved | Notes |
|---|---:|---:|---|---|
| `push_T020_b075_lr0025_J012` | 0.133558 | 1.0 | no | best-push Metropolis: explicit bipolar frozen input, no input internal weights, pattern clamping, Tfactor=0.02, β=0.75, lr=0.0025, relax=500/500, J=0.12 |
| `push_T018_b075_lr0025_J012` | 0.161594 | 1.0 | no | best-push Metropolis: explicit bipolar frozen input, no input internal weights, pattern clamping, Tfactor=0.018, β=0.75, lr=0.0025, relax=500/500, J=0.12 |
| `push_T020_b100_lr0020_J012` | 0.077922 | 1.0 | yes | best-push Metropolis: explicit bipolar frozen input, no input internal weights, pattern clamping, Tfactor=0.02, β=1.0, lr=0.002, relax=500/500, J=0.12 |
| `push_T020_b075_lr0020_J014` | 0.073968 | 1.0 | yes | best-push Metropolis: explicit bipolar frozen input, no input internal weights, pattern clamping, Tfactor=0.02, β=0.75, lr=0.002, relax=500/500, J=0.14 |
| `push_T020_b075_lr0020_J012_rel700` | 0.150357 | 1.0 | no | best-push Metropolis: explicit bipolar frozen input, no input internal weights, pattern clamping, Tfactor=0.02, β=0.75, lr=0.002, relax=700/700, J=0.12 |

Metrics CSV: `checkerboard_4x8_metropolis_best_push_metrics.csv`
Progress PNG: `checkerboard_4x8_metropolis_best_push_progress.png`
