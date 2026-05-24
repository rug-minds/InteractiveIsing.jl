# Inlaid Input Findings

Use of this file: concise notes for the MNIST architecture where the 28x28 pixels are inlaid into a 55x55 layer and separated by live spins.

## Current Diagnostic

- Date: 2026-05-24 03:56:26
- Julia threads: 32
- Architecture: 55x55 input layer with 784 fixed pixel sites, 2241 live separator sites, and 40 output spins.
- Active set: static proposal list excluding only the inlaid pixel sites; this keeps pixels clamped by state without turning off the separator spins.
- Diagnostic output: `C:\Users\fenje\dev\InteractiveIsing.jl\ext\IsingLearning\experiments\MNIST\runs\_diagnostics_inlaid_input\scaling.csv` and `C:\Users\fenje\dev\InteractiveIsing.jl\ext\IsingLearning\experiments\MNIST\runs\_diagnostics_inlaid_input\relaxation.csv`.

## Relaxation Probe

- 10 sweeps: energy mean -644.542, margin mean 0.0, pixels fixed = true.
- 25 sweeps: energy mean -676.353, margin mean 0.333, pixels fixed = true.
- 50 sweeps: energy mean -690.046, margin mean 0.167, pixels fixed = true.
- 75 sweeps: energy mean -693.864, margin mean 0.167, pixels fixed = true.
- 100 sweeps: energy mean -698.346, margin mean 0.333, pixels fixed = true.

## Scaling Probe

- 1 worker: 0.122 s for 8 jobs, 65.53 jobs/s, throughput speedup 1.0x vs 1 worker; ideal 1.0x.
- 8 workers: 0.222 s for 64 jobs, 288.14 jobs/s, throughput speedup 4.4x vs 1 worker; ideal 8.0x.
- 16 workers: 0.246 s for 128 jobs, 519.57 jobs/s, throughput speedup 7.93x vs 1 worker; ideal 16.0x.
- 32 workers: 0.254 s for 256 jobs, 1008.46 jobs/s, throughput speedup 15.39x vs 1 worker; ideal 32.0x.

## Next Run Requirements

- Keep this diagnostic separate from saved training runs.
- Use 32 workers for the actual MNIST runs if scaling remains acceptable.
- Keep pixel sites fixed by the active-set design; do not switch to whole-layer toggling for this architecture.

## Training Result

- `mnist_inlaid_input_training.jl` now has the first useful trainer for this architecture.
- The working recipe uses fixed `[0, 1]` pixel values, live separator spins, pixel-only readout into 40 output replicas, Adam, β=20, unclipped nudged fields through `applied_bias_clip=20`, and fixed output competition between digit groups.
- Saved run: `runs/current/inlaid_pixelreadout_beta20_comp05_lr001_e30_100pc`.
- Result: best and final epoch test accuracy `0.61` on 40 test examples/class; post-hoc best-checkpoint evaluation stayed at `0.61` with 3 reads/100 sweeps and 5 reads/150 sweeps.
