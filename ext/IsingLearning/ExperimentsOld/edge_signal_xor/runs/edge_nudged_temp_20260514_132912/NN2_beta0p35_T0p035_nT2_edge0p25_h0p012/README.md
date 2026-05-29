# Edge Signal XOR

Architecture: 2 input spins, one 8x8 hidden layer, one scalar output spin.
Input spins connect only to the hidden left edge. The output spin connects only to the hidden right edge.

- hidden local NN: `2`
- epochs/log_every: `3000` / `300`
- Minit/eval repeats: `2` / `12`
- free/nudged steps: `900` / `900`
- validation steps: `1800`
- beta/lr/stepsize: `0.35` / `0.0006` / `0.8`
- temperature fraction of max column interaction: `0.035`
- response skipped: `true`

Best logged learning result: epoch `2100`, MSE `0.897269`, accuracy `0.75`.

Files:
- `learning_metrics.csv`
- `learning_progress.png`
- `initial_graph.jld2`
- `best_graph.jld2`
