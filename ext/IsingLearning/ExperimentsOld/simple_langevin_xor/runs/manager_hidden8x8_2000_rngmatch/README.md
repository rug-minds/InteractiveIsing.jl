# Manager-Backed 2 -> 8x8 -> 1 Scalar XOR

This is the same recipe as `hidden8x8_2_64_1.jl`, but EqProp
training trajectories and validation trajectories are dispatched
through persistent `ProcessManager` worker slots.

- workers: `8`
- epochs/log_every: `2000` / `250`
- free/nudged: `600` / `600`
- Minit/eval repeats: `8` / `16`
- β/lr/T/stepsize: `2.0` / `0.003` / `0.005` / `0.4`
- hidden local NN: `1`
- scales input-hidden/local/hidden-output/bias: `0.06` / `0.04` / `0.06` / `0.02`

Best logged: epoch `1250`, MSE `1.192858`, accuracy `0.5`.

CSV: `metrics.csv`
Plot: `progress.png`
Best graph: `hidden8x8_2_64_1_manager_best_graph.jld2`
