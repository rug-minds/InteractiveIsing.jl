# Local Checkerboard Stabilized Search

This run is isolated from the toolbox code. It reuses the checkerboard graph/trainer helpers but replaces the annealed sampler wrapper with no-anneal Processes composites for this search.

## Brainstormed Fixes Tested

- Metropolis no-anneal control, because the previous best local checkerboard runs were discrete Metropolis and the annealing wrapper currently breaks that path.
- Post-update weight clipping, to keep the system in a temperature/coupling regime where states are neither frozen nor noise dominated.
- Optional local-field normalization, to couple the maximum local interaction scale to the chosen temperature.
- Bias suppression/clipping probes, because strong biases can solve single cases while hurting XOR symmetry.

## Results

| Config | Best MSE | Best Acc | Saved | Notes |
|---|---:|---:|---|---|
| `metro_oldrecipe_input_kick` | 0.685547 | 0.75 | no | initially write inactive input checkerboard sites to -1, but keep them unfrozen |

Metrics CSV: `local_checkerboard_stabilized_metrics.csv`

Progress PNG: `local_checkerboard_stabilized_progress.png`
