# Analytic 2->4->1 Scalar XOR

Initialization: `analytic`
The analytic construction uses hidden corner detectors `sign(a*x1 + b*x2 - 1)` and output signs `[-,+,+,-]`.

- `T = 0.02`
- `stepsize = 0.8`
- `β = 1.0`
- target scale = `1.0`
- free/nudged = `1200` / `1200`
- `Minit = 1`, eval repeats `32`

CSV: `metrics.csv`
Plot: `progress.png`
