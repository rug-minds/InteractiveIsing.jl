# Checkerboard 4x8 LocalLangevin Probe

Topology: `4x4 input -> 8x8 hidden -> 4x4 output`. Input is fully frozen bipolar checkerboard embedding; input layer has no internal weights; LocalLangevin uses `adjusted=false`; Hamiltonian terms are asserted to be only `Bilinear`, `MagField`, and output clamping.

| Config | Best MSE | Best Acc | Saved | Notes |
|---|---:|---:|---|---|
| `local_T025_s005_b10_rel500_1000` | 0.982397 | 1.0 | no | LocalLangevin adjusted=false, explicit bipolar frozen input, no input internal weights, Tfactor=0.025, stepsize=0.05, β=1.0, relax=500/1000 |
| `local_T05_s005_b10_rel500_1000` | 0.958401 | 1.0 | no | LocalLangevin adjusted=false, explicit bipolar frozen input, no input internal weights, Tfactor=0.05, stepsize=0.05, β=1.0, relax=500/1000 |
| `local_T025_s010_b10_rel500_1000` | 0.79955 | 1.0 | no | LocalLangevin adjusted=false, explicit bipolar frozen input, no input internal weights, Tfactor=0.025, stepsize=0.1, β=1.0, relax=500/1000 |
| `local_T05_s010_b10_rel500_1000` | 0.790649 | 1.0 | no | LocalLangevin adjusted=false, explicit bipolar frozen input, no input internal weights, Tfactor=0.05, stepsize=0.1, β=1.0, relax=500/1000 |
| `local_T025_s005_b05_rel300_600` | 0.928328 | 1.0 | no | LocalLangevin adjusted=false, explicit bipolar frozen input, no input internal weights, Tfactor=0.025, stepsize=0.05, β=0.5, relax=300/600 |

Metrics CSV: `checkerboard_4x8_local_langevin_metrics.csv`
Progress PNG: `checkerboard_4x8_local_langevin_progress.png`
