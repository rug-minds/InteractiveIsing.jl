# Radius 10 To 1 Learning Grid

This series reruns the single-hidden local MNIST radius grid after the manager
and checkpoint fixes.

- architecture: inactive input layer `784`, sampled `784 -> 121 -> 40`
- radii: `10,9,8,7,6,5,4,3,2,1`
- initialization: scratch unless an interrupted radius is resumed from its own
  `latest_checkpoint.bin`
- dynamics: Metropolis
- free/nudge sweeps: `25` / `25`
- free/nudge reads: `3` / `3`
- beta: `1.5`
- optimizer: Adam
- learning rates W0/W12/W2O/B: `2.5e-5`, `2.5e-5`, `2.5e-5`, `2.5e-6`
- batchsize: `128`
- job chunk size: `8`
- workers: `16`
- train/test per class: `100` / `20`
- epochs: `200`
- launcher: `_launchers/launch_radius_r10_to_r1_resume_safe.ps1`

Resume rule: if a radius has `latest_checkpoint.bin` and its `metrics.csv`
last epoch is below `200`, relaunching the script resumes that radius from the
checkpoint epoch plus one. It does not restart from epoch zero.
