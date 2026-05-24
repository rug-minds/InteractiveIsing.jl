# Field-Scaled Langevin Checkerboard XOR

This run normalizes random edge scales by measured fanout, sets `temp = factor * max_local_interaction_energy(graph)`, and checks adjacency symmetry for every saved graph.

## Results
- `field_global4_langevin_F0p6_R0p25_Tf0p02_eta0p05`: best mse=1.012157, acc=0.75, Teff=0.1873, symmetry=0.0

## Files
- Metrics: `field_scaled_langevin_checkerboard_metrics.csv`
- Summary: `field_scaled_langevin_checkerboard_summary.csv`
- Plot: `field_scaled_langevin_checkerboard_progress.png`
