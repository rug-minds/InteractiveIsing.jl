# Edge Signal XOR

Architecture: 2 input spins, one 8x8 hidden layer, one scalar output spin.
Input spins connect only to the hidden left edge. The output spin connects only to the hidden right edge.

- hidden local NN: `2`
- epochs/log_every: `600` / `100`
- Minit/eval repeats: `4` / `8`
- free/nudged steps: `500` / `500`
- validation steps: `3000`
- beta/lr/stepsize: `2.0` / `0.002` / `0.4`
- temperature fraction of max column interaction: `0.025`
- response skipped: `true`

Best logged learning result: epoch `1`, MSE `0.80522`, accuracy `0.75`.

Files:
- `learning_metrics.csv`
- `learning_progress.png`
- `initial_graph.jld2`
- `best_graph.jld2`
