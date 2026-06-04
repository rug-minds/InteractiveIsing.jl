# R8 Rescue Continuation

Created: 2026-06-04.

This short-path series continues runs that were interrupted or vulnerable to
Windows checkpoint path-length issues in `20260604_r8_rescue_series`.

- `a15r`: resume `a15_s50_b10_lr10_e200` from its latest checkpoint and run
  another 76 epochs with the same `50/50`, `β=1.0`, `lr=1e-5` settings.
- `a16`: run `50/50`, `β=0.75`, `lr=1e-5` from scratch.

Both runs use r8, Metropolis, batch size 128, chunk size 4, and 32 workers.
