# Checkerboard 4x8 Tight Metropolis Probe

This run tightens the search around the corrected checkerboard recipe that first reached MSE near `0.13`: fully frozen bipolar input, no input-layer internal weights, pattern output clamping, no polynomial/double-well local potentials, symmetric Metropolis dynamics.

| Config | Best MSE | Best Acc | Saved | Notes |
|---|---:|---:|---|---|
| `metro_T020_b05_lr003_rel300_J010` | 0.201701 | 1.0 | no | tight Metropolis probe: explicit bipolar frozen input, no input internal weights, pattern clamping, Tfactor=0.02, β=0.5, lr=0.003, relax=300/300, J=0.1, hidden/output internal=0.06/0.06 |
| `metro_T025_b05_lr003_rel300_J010` | 0.207987 | 1.0 | no | tight Metropolis probe: explicit bipolar frozen input, no input internal weights, pattern clamping, Tfactor=0.025, β=0.5, lr=0.003, relax=300/300, J=0.1, hidden/output internal=0.06/0.06 |
| `metro_T030_b05_lr003_rel300_J010` | 0.208381 | 1.0 | no | tight Metropolis probe: explicit bipolar frozen input, no input internal weights, pattern clamping, Tfactor=0.03, β=0.5, lr=0.003, relax=300/300, J=0.1, hidden/output internal=0.06/0.06 |
| `metro_T025_b075_lr0025_rel300_J010` | 0.226318 | 1.0 | no | tight Metropolis probe: explicit bipolar frozen input, no input internal weights, pattern clamping, Tfactor=0.025, β=0.75, lr=0.0025, relax=300/300, J=0.1, hidden/output internal=0.06/0.06 |
| `metro_T025_b05_lr0025_rel500_J010` | 0.225545 | 1.0 | no | tight Metropolis probe: explicit bipolar frozen input, no input internal weights, pattern clamping, Tfactor=0.025, β=0.5, lr=0.0025, relax=500/500, J=0.1, hidden/output internal=0.06/0.06 |
| `metro_T025_b05_lr003_rel300_J012` | 0.185316 | 1.0 | no | tight Metropolis probe: explicit bipolar frozen input, no input internal weights, pattern clamping, Tfactor=0.025, β=0.5, lr=0.003, relax=300/300, J=0.12, hidden/output internal=0.06/0.06 |
| `metro_T025_b05_lr003_rel300_internal008` | 0.259786 | 1.0 | no | tight Metropolis probe: explicit bipolar frozen input, no input internal weights, pattern clamping, Tfactor=0.025, β=0.5, lr=0.003, relax=300/300, J=0.1, hidden/output internal=0.08/0.08 |
| `metro_T020_b075_lr0025_rel500_J012` | 0.116957 | 1.0 | yes | tight Metropolis probe: explicit bipolar frozen input, no input internal weights, pattern clamping, Tfactor=0.02, β=0.75, lr=0.0025, relax=500/500, J=0.12, hidden/output internal=0.06/0.06 |

Metrics CSV: `checkerboard_4x8_metropolis_tight_metrics.csv`
Progress PNG: `checkerboard_4x8_metropolis_tight_progress.png`
