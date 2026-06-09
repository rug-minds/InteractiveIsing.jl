# Fine Tuning From R5 And R10 Best Checkpoints

This folder is for continuation experiments only. The parent radius grid remains
unchanged.

Starting checkpoints:

- r5: parent `r5/best_params.bin`, best epoch `138`, best test accuracy `0.500`
- r10: parent `r10/best_params.bin`, best epoch `151`, best test accuracy `0.495`

All fine-tuning runs initialize from those best checkpoints and preserve the
optimizer state stored in the checkpoint. The target epoch count is absolute:

- r5 runs continue from epoch `139` through epoch `238`
- r10 runs continue from epoch `152` through epoch `251`

Branches:

- `same_sampler_lr10`: keep `25/25` sweeps and `beta=1.5`, lower Adam LR to
  `1e-5` (`b=1e-6`).
- `beta10_lr10`: keep `25/25` sweeps, reduce `beta=1.0`, Adam LR `1e-5`.
- `s35_beta10_lr10`: increase relaxation to `35/35` sweeps, `beta=1.0`, Adam
  LR `1e-5`.

Launcher: `_launchers/launch_fine_tune_r5_r10.ps1`.

Resume rule: relaunching the launcher skips completed runs and resumes any
partial run from its own `latest_checkpoint.bin` at the next epoch.
