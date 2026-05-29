# Low-Beta Edge Signal XOR Search

This run keeps the edge architecture fixed and tests the scalar-XOR lesson that full `±1` targets need a smaller direct clamping `β` than the old edge runs used.

| trial | NN | beta | T fraction | edge scale | hidden scale | best MSE | best acc | best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `NN2_beta0p20_T0p05_edge0p20_h0p015` | 2 | 0.2 | 0.05 | 0.2 | 0.015 | 0.781474 | 1.0 | 720 |
| `NN3_beta0p35_T0p035_edge0p25_h0p010` | 3 | 0.35 | 0.035 | 0.25 | 0.01 | 0.838903 | 0.75 | 1 |
| `NN2_beta0p35_T0p05_edge0p20_h0p015` | 2 | 0.35 | 0.05 | 0.2 | 0.015 | 0.873034 | 0.75 | 2400 |
| `NN2_beta0p20_T0p035_edge0p25_h0p012` | 2 | 0.2 | 0.035 | 0.25 | 0.012 | 1.095823 | 0.75 | 720 |
| `NN3_beta0p20_T0p05_edge0p25_h0p010` | 3 | 0.2 | 0.05 | 0.25 | 0.01 | 1.230901 | 0.75 | 1680 |
| `NN1_beta0p20_T0p05_edge0p20_h0p020` | 1 | 0.2 | 0.05 | 0.2 | 0.02 | 0.964609 | 0.5 | 1680 |

The success criterion is accuracy `1.0` and scalar-output MSE below `0.1` from repeated validation starts.
