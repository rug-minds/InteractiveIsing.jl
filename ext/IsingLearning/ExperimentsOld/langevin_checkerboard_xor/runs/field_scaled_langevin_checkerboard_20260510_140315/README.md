# Field-Scaled Langevin Checkerboard XOR

This run normalizes random edge scales by measured fanout, sets `temp = factor * max_local_interaction_energy(graph)`, and checks adjacency symmetry for every saved graph.

## Results
- `field_global4_langevin_F0p6_R0p25_Tf0p02_eta0p05`: best mse=0.917638, acc=0.75, Teff=0.1873, symmetry=0.0
- `field_global4_langevin_F0p3_R0p25_Tf0p02_eta0p05`: best mse=0.931439, acc=0.75, Teff=0.09367, symmetry=0.0
- `field_global4_langevin_F0p6_R0p25_Tf0p005_eta0p05`: best mse=0.946722, acc=0.75, Teff=0.04683, symmetry=0.0
- `field_global4_langevin_F0p3_R0p25_Tf0p005_eta0p05`: best mse=0.947757, acc=0.75, Teff=0.02342, symmetry=0.0
- `field_global4_global_langevin_F0p3_R0p25_Tf0p005_eta0p05`: best mse=0.960071, acc=0.75, Teff=0.02342, symmetry=0.0
- `field_global4_global_langevin_F0p3_R0p25_Tf0p02_eta0p05`: best mse=0.964278, acc=0.75, Teff=0.09367, symmetry=0.0
- `field_global4_global_langevin_F0p6_R0p25_Tf0p005_eta0p05`: best mse=0.970997, acc=0.75, Teff=0.04683, symmetry=0.0
- `field_global4_global_langevin_F0p6_R0p25_Tf0p02_eta0p05`: best mse=0.989699, acc=0.75, Teff=0.1873, symmetry=0.0

## Files
- Metrics: `field_scaled_langevin_checkerboard_metrics.csv`
- Summary: `field_scaled_langevin_checkerboard_summary.csv`
- Plot: `field_scaled_langevin_checkerboard_progress.png`
