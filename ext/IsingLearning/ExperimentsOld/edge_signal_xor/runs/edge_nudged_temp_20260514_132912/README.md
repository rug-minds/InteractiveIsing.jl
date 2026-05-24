# Edge Signal XOR With Nudged Temperature Bump

The free phase runs at the configured base temperature. The plus/minus nudged phases run at `nudged_temp_factor * base_temperature`, then restore the base temperature.

| trial | NN | beta | T fraction | nudged T factor | edge scale | hidden scale | best MSE | best acc | best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `NN2_beta0p20_T0p04_nT4_edge0p25_h0p012` | 2 | 0.2 | 0.04 | 4.0 | 0.25 | 0.012 | 0.684345 | 1.0 | 0 |
| `NN3_beta0p20_T0p035_nT3_edge0p30_h0p008` | 3 | 0.2 | 0.035 | 3.0 | 0.3 | 0.008 | 0.684721 | 0.75 | 300 |
| `NN2_beta0p35_T0p035_nT2_edge0p25_h0p012` | 2 | 0.35 | 0.035 | 2.0 | 0.25 | 0.012 | 0.897269 | 0.75 | 2100 |
| `NN2_beta0p20_T0p04_nT2_edge0p25_h0p012` | 2 | 0.2 | 0.04 | 2.0 | 0.25 | 0.012 | 0.906753 | 0.75 | 1200 |
