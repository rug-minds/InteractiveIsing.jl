# High-Sweep Fine Tuning

This folder extends the r5/r10 fine-tuning with genuinely larger relaxation
counts. The earlier fine-tuning suite only tested up to `35/35` sweeps.

Starting checkpoints:

- r5: parent `r5/best_params.bin`, best epoch `138`, best test accuracy `0.500`
- r10: parent `r10/best_params.bin`, best epoch `151`, best test accuracy `0.495`

Branches:

- `s50_beta15_lr10`: `50/50` sweeps, `beta=1.5`, Adam LR `1e-5`, 100 extra epochs.
- `s50_beta10_lr10`: `50/50` sweeps, `beta=1.0`, Adam LR `1e-5`, 100 extra epochs.
- `s75_beta10_lr5`: `75/75` sweeps, `beta=1.0`, Adam LR `5e-6`, 80 extra epochs.

Launcher: `_launchers/launch_high_sweep_fine_tune_r5_r10.ps1`.

Resume rule: relaunching skips completed runs and resumes partial runs from
their own `latest_checkpoint.bin`.
