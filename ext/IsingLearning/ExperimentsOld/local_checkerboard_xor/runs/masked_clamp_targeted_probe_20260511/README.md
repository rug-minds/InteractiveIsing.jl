# Targeted Masked-Clamp Checkerboard Probe

Follow-up to the first masked-clamp beta probe. The first probe favored output-pattern clamping over scalar readout clamping, so this run concentrates on high-β output-pattern Langevin points.

| Config | Best MSE | Best Acc | Saved | Notes |
|---|---:|---:|---|---|
| `block_pattern_b15_T02_r300_900` | 0.810156 | 0.75 | no | targeted masked clamp follow-up: langevin, output-pattern clamp, β=1.5, Tfactor=0.02, relax=300/900, J=0.2 |
| `block_pattern_b20_T02_r300_900` | 0.829756 | 0.75 | no | targeted masked clamp follow-up: langevin, output-pattern clamp, β=2.0, Tfactor=0.02, relax=300/900, J=0.2 |
| `block_pattern_b15_T03_r300_900` | 0.740986 | 0.75 | no | targeted masked clamp follow-up: langevin, output-pattern clamp, β=1.5, Tfactor=0.03, relax=300/900, J=0.2 |
| `block_pattern_b10_T015_r500_1000` | 0.678949 | 0.75 | no | targeted masked clamp follow-up: langevin, output-pattern clamp, β=1.0, Tfactor=0.015, relax=500/1000, J=0.2 |
| `block_pattern_b15_T02_J030_r300_900` | 0.792751 | 0.75 | no | targeted masked clamp follow-up: langevin, output-pattern clamp, β=1.5, Tfactor=0.02, relax=300/900, J=0.3 |
| `global_pattern_b15_T02_r300_900` | 0.668761 | 0.75 | no | targeted masked clamp follow-up: global_langevin, output-pattern clamp, β=1.5, Tfactor=0.02, relax=300/900, J=0.2 |

Metrics CSV: `masked_clamp_targeted_metrics.csv`
Progress PNG: `masked_clamp_targeted_progress.png`
