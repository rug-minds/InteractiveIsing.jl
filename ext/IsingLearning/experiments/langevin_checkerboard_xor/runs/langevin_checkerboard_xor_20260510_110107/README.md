# Langevin Checkerboard XOR

This run tests physical checkerboard input masks with continuous `BlockLangevin` dynamics.
The input is not four-case one-hot: bit A freezes one checkerboard mask to `+1`, bit B freezes the complementary mask to `+1`.

## Summary
- `lgv_8x8_global8`: best mse=0.987141, best acc=0.5, final mse=1.010013, final acc=0.75
- `lgv_8x8_inlaid4`: best mse=0.995029, best acc=0.5, final mse=1.004051, final acc=0.5

## Files
- Metrics: `langevin_checkerboard_xor_metrics.csv`
- Summary: `langevin_checkerboard_xor_summary.csv`
- Plot: `langevin_checkerboard_xor_progress.png`
- Per-config folders contain best graph JLD2 files and parameter SVGs.
