# Edge Signal XOR

Architecture: 2 input spins, one 8x8 hidden layer, one scalar output spin.
Input spins connect only to the hidden left edge. The output spin connects only to the hidden right edge.

- hidden local NN: `5`
- epochs/log_every: `1200` / `120`
- Minit/eval repeats: `6` / `16`
- free/nudged steps: `300` / `300`
- validation steps: `600`
- beta/lr/stepsize: `1.0` / `0.0008` / `0.8`
- temperature fraction of max column interaction: `0.05`
- response skipped: `true`

Best logged learning result: epoch `0`, MSE `0.792204`, accuracy `0.75`.

Files:
- `learning_metrics.csv`
- `learning_progress.png`
- `initial_graph.jld2`
- `best_graph.jld2`
