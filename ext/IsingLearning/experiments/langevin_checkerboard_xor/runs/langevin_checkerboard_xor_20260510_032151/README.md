# Langevin Checkerboard XOR

This run tests physical checkerboard input masks with continuous `BlockLangevin` dynamics.
The input is not four-case one-hot: bit A freezes one checkerboard mask to `+1`, bit B freezes the complementary mask to `+1`.

## Summary
- `lgv_4x4_global4`: best mse=0.926772, best acc=0.75, final mse=0.926885, final acc=0.75

## Files
- Metrics: `langevin_checkerboard_xor_metrics.csv`
- Summary: `langevin_checkerboard_xor_summary.csv`
- Plot: `langevin_checkerboard_xor_progress.png`
- Per-config folders contain best graph JLD2 files and parameter SVGs.
