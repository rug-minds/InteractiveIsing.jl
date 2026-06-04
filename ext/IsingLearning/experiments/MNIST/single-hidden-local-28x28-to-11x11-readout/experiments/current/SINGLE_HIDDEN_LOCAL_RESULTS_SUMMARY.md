# Single-Hidden-Local MNIST Results Summary

Generated: 2026-06-04

## Verdict

No saved run was successful. The strongest run only reached a transient best
test accuracy of `0.275` and finished at chance. Most runs collapsed to stable
single-class, chance-level prediction around `0.10`.

The folders that produced no usable metrics or only failed launch state were
also treated as unsuccessful.

## What Worked At All

| Run family | Best setting | Best test acc. | Final test acc. | Notes |
|---|---:|---:|---:|---|
| `r1_to_r9_relax_grid_e30_32w` | `r=3`, `sweeps=50`, `beta=5` | `0.275` | `0.095` | Weak transient learning, then chance. |
| `r1_to_r9_relax_grid_e30_32w` | `r=3`, `sweeps=25`, `beta=5` | `0.240` | `0.080` | Same pattern, weaker. |
| `r8_low_lr_no_output_bias_relax_beta_grid_e30_32w` | `r=8`, `sweeps=25`, `beta=10`, low LR, no output bias | `0.240` | `0.100` | Rescue attempt gave transient bump only. |
| `r1_to_r9_relax_grid_e30_32w` | `r=4`, `sweeps=25`, `beta=5` | `0.190` | `0.100` | Did not sustain. |
| `r1_to_r9_relax_grid_e30_32w` | `r=5`, `sweeps=25`, `beta=5` | `0.180` | `0.100` | Did not sustain. |

The only region with visible non-chance behavior was small radius, especially
`r=3`. Even there, the result did not remain above chance by the end of the run.

## What Did Not Work

- `r8` did not learn in the original beta/relaxation grid. Best saved value was
  `0.135`, with final accuracy at `0.10`.
- Increasing `r8` relaxation sweeps to `150`, `250`, `400`, `800`, or `1200`
  did not help. Higher sweeps often early-stopped as collapsed single-class
  prediction.
- Removing output bias did not fix the collapse. Best no-output-bias `r8`
  result was `0.135`.
- Lowering learning rate plus no output bias improved only transiently; best
  saved `r8` result was `0.240`, final still `0.10`.
- The `r=8` Metropolis-from-scratch run reached only `0.115`.
- The short `r=3` two-epoch probes were not successful; best was `0.160`.
- The `r1_to_r10_low_lr_no_output_bias_sweep_grid_e60_32w` and
  `r4_rescue_lr_beta_sweep_grid_e80_32w` folders contained launch/log files but
  no usable metric CSVs.

## Cleanup

All generated run folders from this experiment set were removed because none
showed sustained learning. This summary is the retained record of the saved
results.
