# Analytic 2->4->1 Scalar XOR

Initialization: `analytic`
The analytic construction uses hidden corner detectors `sign(a*x1 + b*x2 - 1)` and output signs `[-,+,+,-]`.

- `T = 0.005`
- `stepsize = 0.4`
- `β = 2.0`
- target scale = `1.0`
- free/nudged = `600` / `600`
- `Minit = 1`, eval repeats `64`

CSV: `metrics.csv`
Plot: `progress.png`
