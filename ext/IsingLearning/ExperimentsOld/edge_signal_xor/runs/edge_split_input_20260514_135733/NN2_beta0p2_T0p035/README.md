# Edge Signal XOR

Architecture: 2 input spins, one 8x8 hidden layer, one scalar output spin.
Input spins connect only to the hidden left edge. The output spin connects only to the hidden right edge.

- hidden local NN: `2`
- epochs/log_every: `6000` / `600`
- Minit/eval repeats: `2` / `12`
- free/nudged steps: `1000` / `1000`
- validation steps: `2500`
- beta/lr/stepsize: `0.2` / `0.0007` / `0.8`
- temperature fraction of max column interaction: `0.035`
- response skipped: `true`

Best logged learning result: epoch `5400`, MSE `1.194949`, accuracy `0.5`.

Files:
- `learning_metrics.csv`
- `learning_progress.png`
- `initial_graph.jld2`
- `best_graph.jld2`
