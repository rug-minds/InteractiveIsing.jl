# 2 -> 8x8 -> 1 Scalar XOR

Architecture: all-to-all input-hidden, hidden NN=1 local couplings, all-to-one hidden-output.

- epochs/log_every: `600` / `200`
- free/nudged: `300` / `300`
- Minit/eval repeats: `4` / `8`
- β/lr/T/stepsize: `2.0` / `0.002` / `0.005` / `0.4`
- scales input-hidden/local/hidden-output/bias: `0.06` / `0.04` / `0.06` / `0.02`

Best logged: epoch `600`, MSE `0.653001`, accuracy `0.75`.

CSV: `metrics.csv`
Plot: `progress.png`
