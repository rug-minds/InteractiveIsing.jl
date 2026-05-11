# Masked Clamping Beta Probe

Focused checkerboard XOR probes after adding a clamping mask.

| Config | Best MSE | Best Acc | Saved | Notes |
|---|---:|---:|---|---|
| `block_readout_b05_T02_r150_300` | 0.969143 | 0.75 | no | masked-clamping beta probe: langevin, clamp=readout, β=0.5, Tfactor=0.02, relax=150/300 |
| `block_readout_b10_T02_r150_300` | 0.914896 | 0.75 | no | masked-clamping beta probe: langevin, clamp=readout, β=1.0, Tfactor=0.02, relax=150/300 |
| `block_readout_b20_T02_r150_300` | 0.91745 | 0.75 | no | masked-clamping beta probe: langevin, clamp=readout, β=2.0, Tfactor=0.02, relax=150/300 |
| `block_readout_b10_T05_r150_300` | 0.950306 | 0.75 | no | masked-clamping beta probe: langevin, clamp=readout, β=1.0, Tfactor=0.05, relax=150/300 |
| `block_pattern_b10_T02_r300_600` | 0.696578 | 0.75 | no | masked-clamping beta probe: langevin, clamp=pattern, β=1.0, Tfactor=0.02, relax=300/600 |
| `global_readout_b10_T02_r150_300` | 0.825742 | 0.75 | no | masked-clamping beta probe: global_langevin, clamp=readout, β=1.0, Tfactor=0.02, relax=150/300 |

Metrics CSV: `masked_clamp_beta_probe_metrics.csv`
Progress PNG: `masked_clamp_beta_probe_progress.png`
