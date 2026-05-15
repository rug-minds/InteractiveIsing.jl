# Edge Signal XOR

Architecture: 2 input spins, one 8x8 hidden layer, one scalar output spin.
Input spins connect only to the hidden left edge. The output spin connects only to the hidden right edge.

- hidden local NN: `1`
- epochs/log_every: `3000` / `300`
- Minit/eval repeats: `2` / `16`
- free/nudged steps: `900` / `900`
- validation steps: `1800`
- beta/lr/stepsize: `0.35` / `0.0007` / `0.8`
- temperature fraction of max column interaction: `0.04`
- response skipped: `true`

Best logged learning result: epoch `1`, MSE `1.074729`, accuracy `0.5`.

Files:
- `learning_metrics.csv`
- `learning_progress.png`
- `initial_graph.jld2`
- `best_graph.jld2`
