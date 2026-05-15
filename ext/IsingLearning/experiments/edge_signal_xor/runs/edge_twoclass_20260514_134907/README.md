# Edge Signal XOR With Two-Class Output Edge

The input remains two raw spins connected only to the hidden left edge. The output is an 8x2 layer. Classification uses `mean(true column) - mean(false column)`.

| trial | NN | beta | T fraction | input scale | hidden scale | output scale | best MSE | best acc | best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `NN5_beta0p2_T0p025` | 5 | 0.2 | 0.025 | 0.35 | 0.005 | 0.4 | 3.472601 | 1.0 | 4500 |
| `NN3_beta0p2_T0p03` | 3 | 0.2 | 0.03 | 0.3 | 0.009 | 0.35 | 3.93569 | 1.0 | 1000 |
| `NN2_beta0p2_T0p035` | 2 | 0.2 | 0.035 | 0.25 | 0.014 | 0.3 | 3.8376 | 0.75 | 1000 |
