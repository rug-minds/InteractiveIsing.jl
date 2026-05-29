# 2 -> 8x8 -> 1 Scalar XOR

Architecture: all-to-all input-hidden, hidden NN=1 local couplings, all-to-one hidden-output.

- epochs/log_every: `2500` / `500`
- free/nudged: `600` / `600`
- Minit/eval repeats: `8` / `16`
- β/lr/T/stepsize: `2.0` / `0.002` / `0.005` / `0.4`
- hidden local NN: `1`
- scales input-hidden/local/hidden-output/bias: `0.06` / `0.04` / `0.06` / `0.02`

Best logged: epoch `1000`, MSE `0.450048`, accuracy `1.0`.

CSV: `metrics.csv`
Plot: `progress.png`
Best graph: `hidden8x8_2_64_1_best_graph.jld2`
