# Analytic 2->4->1 Scalar XOR

Initialization: `random`
The analytic construction uses hidden corner detectors `sign(a*x1 + b*x2 - 1)` and output signs `[-,+,+,-]`.

- `T = 0.005`
- `stepsize = 0.4`
- `β = 2.0`
- target scale = `1.0`
- learning rate = `0.005`
- gradient sign = `1.0`
- random weight scale = `0.15`
- random bias scale = `0.05`
- free/nudged = `600` / `600`
- `Minit = 8`, eval repeats `16`

CSV: `metrics.csv`
Plot: `progress.png`
