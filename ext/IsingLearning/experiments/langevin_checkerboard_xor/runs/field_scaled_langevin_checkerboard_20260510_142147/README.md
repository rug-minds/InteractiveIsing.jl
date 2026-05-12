# Field-Scaled Langevin Checkerboard XOR

This run normalizes random edge scales by measured fanout, sets `temp = factor * max_local_interaction_energy(graph)`, and checks adjacency symmetry for every saved graph.

## Results
- `field_global4_global_langevin_F1p5_R0p0_Tf0p02_eta0p08`: best mse=0.957837, acc=0.75, Teff=0.404, symmetry=0.0
- `field_global4_langevin_F1p0_R0p1_Tf0p02_eta0p08`: best mse=0.964435, acc=0.75, Teff=0.2865, symmetry=0.0
- `field_global4_global_langevin_F1p5_R0p1_Tf0p02_eta0p08`: best mse=0.965249, acc=0.5, Teff=0.4297, symmetry=0.0
- `field_global4_global_langevin_F1p0_R0p1_Tf0p02_eta0p08`: best mse=0.974216, acc=0.5, Teff=0.2865, symmetry=0.0
- `field_global4_langevin_F1p5_R0p1_Tf0p02_eta0p08`: best mse=0.975586, acc=0.75, Teff=0.4297, symmetry=0.0
- `field_global4_langevin_F1p5_R0p0_Tf0p02_eta0p08`: best mse=0.977298, acc=0.5, Teff=0.404, symmetry=0.0
- `field_global4_langevin_F1p0_R0p0_Tf0p02_eta0p08`: best mse=0.977628, acc=0.75, Teff=0.2693, symmetry=0.0
- `field_global4_global_langevin_F1p0_R0p1_Tf0p005_eta0p08`: best mse=0.989392, acc=0.75, Teff=0.07162, symmetry=0.0
- `field_global4_global_langevin_F1p0_R0p0_Tf0p005_eta0p08`: best mse=0.989498, acc=0.75, Teff=0.06733, symmetry=0.0
- `field_global4_global_langevin_F1p0_R0p0_Tf0p02_eta0p08`: best mse=1.002945, acc=0.5, Teff=0.2693, symmetry=0.0
- `field_global4_global_langevin_F1p5_R0p0_Tf0p005_eta0p08`: best mse=1.007347, acc=0.5, Teff=0.101, symmetry=0.0
- `field_global4_langevin_F1p0_R0p0_Tf0p005_eta0p08`: best mse=1.009427, acc=0.75, Teff=0.06733, symmetry=0.0
- `field_global4_langevin_F1p0_R0p1_Tf0p005_eta0p08`: best mse=1.01624, acc=0.75, Teff=0.07162, symmetry=0.0
- `field_global4_langevin_F1p5_R0p0_Tf0p005_eta0p08`: best mse=1.029085, acc=0.75, Teff=0.101, symmetry=0.0
- `field_global4_langevin_F1p5_R0p1_Tf0p005_eta0p08`: best mse=1.039852, acc=0.75, Teff=0.1074, symmetry=0.0
- `field_global4_global_langevin_F1p5_R0p1_Tf0p005_eta0p08`: best mse=1.047754, acc=0.75, Teff=0.1074, symmetry=0.0

## Files
- Metrics: `field_scaled_langevin_checkerboard_metrics.csv`
- Summary: `field_scaled_langevin_checkerboard_summary.csv`
- Plot: `field_scaled_langevin_checkerboard_progress.png`
