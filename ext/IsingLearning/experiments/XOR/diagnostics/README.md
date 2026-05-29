# XOR Diagnostics

This folder is for non-result runs: smoke tests, sanity checks, scouting runs,
and manager timing diagnostics. Keep these separate from architecture result
grids so failed probes do not pollute the checkerboard or edge comparison trees.

The timing entry point is `xor_local_manager_diagnostics.jl`. It imports the
checkerboard CNN-like experiment and times a small number of manager batches
while recording worker sharing diagnostics.

- `runs`: raw diagnostic outputs and generated diagnostic plots.
