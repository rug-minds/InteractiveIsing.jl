# Field-Scaled Langevin Checkerboard XOR

This run normalizes random edge scales by measured fanout, sets `temp = factor * max_local_interaction_energy(graph)`, and checks adjacency symmetry for every saved graph.

## Results
- `field_global4_global_langevin_F0p3_R0p25_Tf0p01_eta0p05`: best mse=0.991086, acc=0.75, Teff=0.04683, symmetry=0.0

## Files
- Metrics: `field_scaled_langevin_checkerboard_metrics.csv`
- Summary: `field_scaled_langevin_checkerboard_summary.csv`
- Plot: `field_scaled_langevin_checkerboard_progress.png`
