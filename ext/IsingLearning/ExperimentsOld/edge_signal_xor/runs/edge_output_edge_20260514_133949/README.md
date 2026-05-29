# Edge Signal XOR With Output Edge Readout

This keeps two physical input spins on the left edge path, but uses eight output spins and classifies by their mean sign.

| trial | NN | beta | T fraction | edge scale | hidden scale | best MSE | best acc | best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `NN3_beta0p2_T0p035` | 3 | 0.2 | 0.035 | 0.25 | 0.01 | 0.998847 | 0.75 | 2000 |
| `NN2_beta0p35_T0p025` | 2 | 0.35 | 0.025 | 0.25 | 0.012 | 1.007292 | 0.5 | 1200 |
| `NN2_beta0p2_T0p035` | 2 | 0.2 | 0.035 | 0.22 | 0.015 | 1.11506 | 0.25 | 0 |
