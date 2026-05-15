# Edge Signal XOR With Split Input Halves

Architecture remains `2 input spins -> left edge of 8x8 hidden -> right edge -> 1 output spin`.
Input spin 1 connects only to the upper half of the first hidden edge. Input spin 2 connects only to the lower half.

| trial | NN | beta | T fraction | input scale | hidden scale | output scale | best MSE | best acc | best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `NN5_beta0p2_T0p025` | 5 | 0.2 | 0.025 | 0.45 | 0.005 | 0.4 | 0.688475 | 0.75 | 1800 |
| `NN3_beta0p2_T0p03` | 3 | 0.2 | 0.03 | 0.4 | 0.009 | 0.35 | 0.864332 | 0.75 | 1 |
| `NN3_beta0p35_T0p025` | 3 | 0.35 | 0.025 | 0.4 | 0.009 | 0.35 | 1.126974 | 0.5 | 1800 |
| `NN2_beta0p2_T0p035` | 2 | 0.2 | 0.035 | 0.35 | 0.014 | 0.3 | 1.194949 | 0.5 | 5400 |
