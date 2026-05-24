# Manager-Backed 2 -> 8x8 -> 1 Scalar XOR

This is the same recipe as `hidden8x8_2_64_1.jl`, but EqProp
training trajectories and validation trajectories are dispatched
through persistent `ProcessManager` worker slots.

- workers: `2`
- epochs/log_every: `2` / `1`
- free/nudged: `20` / `20`
- Minit/eval repeats: `2` / `2`
- β/lr/T/stepsize: `2.0` / `0.002` / `0.005` / `0.4`
- hidden local NN: `1`
- scales input-hidden/local/hidden-output/bias: `0.06` / `0.04` / `0.06` / `0.02`

Best logged: epoch `2`, MSE `1.20821`, accuracy `0.5`.

CSV: `metrics.csv`
Plot: `progress.png`
Best graph: `hidden8x8_2_64_1_manager_best_graph.jld2`
