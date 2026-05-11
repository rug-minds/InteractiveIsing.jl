# Langevin Checkerboard XOR

This run tests physical checkerboard input masks with continuous `BlockLangevin` dynamics.
The input is not four-case one-hot: bit A freezes one checkerboard mask to `+1`, bit B freezes the complementary mask to `+1`.

## Summary
- `lgv_4x4_global4`: best mse=0.333032, best acc=1.0, final mse=0.333032, final acc=1.0

## Files
- Metrics: `langevin_checkerboard_xor_metrics.csv`
- Summary: `langevin_checkerboard_xor_summary.csv`
- Plot: `langevin_checkerboard_xor_progress.png`
- Per-config folders contain best graph JLD2 files and parameter SVGs.
