# Analytic 2->4->1 Scalar XOR

Initialization: `random`
The analytic construction uses hidden corner detectors `sign(a*x1 + b*x2 - 1)` and output signs `[-,+,+,-]`.

- `T = 0.001`
- `stepsize = 0.2`
- `β = 2.0`
- free/nudged = `600` / `600`
- `Minit = 8`, eval repeats `16`

CSV: `metrics.csv`
Plot: `progress.png`
