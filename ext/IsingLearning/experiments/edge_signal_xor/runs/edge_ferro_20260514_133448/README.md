# Edge Signal XOR With Positive Hidden Local Couplings

Hidden-local couplings start positive so the 8x8 hidden layer is a signal-carrying medium instead of a signed random local network. Edge input/output couplings are still signed random and trainable.

| trial | NN | beta | T fraction | edge scale | hidden scale | best MSE | best acc | best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `NN2_beta0p2_T0p035` | 2 | 0.2 | 0.035 | 0.25 | 0.02 | 0.847426 | 0.75 | 0 |
| `NN3_beta0p2_T0p025` | 3 | 0.2 | 0.025 | 0.3 | 0.012 | 1.067709 | 0.5 | 0 |
| `NN1_beta0p2_T0p035` | 1 | 0.2 | 0.035 | 0.22 | 0.035 | 1.295451 | 0.5 | 400 |
