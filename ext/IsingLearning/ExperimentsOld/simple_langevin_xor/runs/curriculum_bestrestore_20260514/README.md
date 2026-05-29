# Scalar 2->4->1 XOR Curriculum

Random initialization, no local potential, unadjusted `LocalLangevin`.
The target is ramped from weak scalar clamping to the full `±1` scalar target.

- T = `0.07`
- stepsize = `0.8`
- beta = `1.0`
- free/nudged = `1200` / `1200`
- Minit / eval repeats = `1` / `24`
- restored best epoch = `4500`
- restored best MSE = `0.027361730474628226`
- restored best accuracy = `1.0`

| stage target scale | epochs | learning rate |
|---:|---:|---:|
| 0.25 | 1200 | 0.001 |
| 0.5 | 1200 | 0.0008 |
| 1.0 | 2400 | 0.0005 |

CSV: `metrics.csv`
Plot: `progress.png`
