# Scalar 2->4->1 XOR Curriculum

Random initialization, no local potential, unadjusted `LocalLangevin`.
The target is ramped from weak scalar clamping to the full `±1` scalar target.

- T = `0.07`
- stepsize = `0.8`
- beta = `1.0`
- free/nudged = `10` / `10`
- Minit / eval repeats = `1` / `2`

| stage target scale | epochs | learning rate |
|---:|---:|---:|
| 0.25 | 1 | 0.001 |
| 0.5 | 1 | 0.0008 |
| 1.0 | 1 | 0.0005 |

CSV: `metrics.csv`
Plot: `progress.png`

## Nudged-Phase Annealing

- start temperature = `1.5*T`
- stop temperature = `T`
- power = `1.0`

Only the plus/minus nudged branches use this schedule. The free and validation
branches use the base `T` from the run config.
