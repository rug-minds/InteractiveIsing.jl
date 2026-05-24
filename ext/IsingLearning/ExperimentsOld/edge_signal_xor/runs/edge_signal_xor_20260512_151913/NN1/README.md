# Edge Signal XOR

Architecture: 2 input spins, one 8x8 hidden layer, one scalar output spin.
Input spins connect only to the hidden left edge. The output spin connects only to the hidden right edge.

- hidden local NN: `1`
- epochs/log_every: `500` / `100`
- Minit/eval repeats: `4` / `4`
- free/nudged steps: `400` / `400`
- beta/lr/stepsize: `2.0` / `0.003` / `0.4`
- temperature fraction of max column interaction: `0.025`

Best logged learning result: epoch `200`, MSE `0.856821`, accuracy `0.75`.

Files:
- `learning_metrics.csv`
- `learning_progress.png`
- `response_rows.csv`
- `response_profiles.csv`
- `response_summary.png`
- `initial_graph.jld2`
- `best_graph.jld2`
