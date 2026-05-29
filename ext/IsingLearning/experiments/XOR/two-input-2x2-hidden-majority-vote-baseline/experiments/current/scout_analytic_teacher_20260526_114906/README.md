# Two-Input 2x2 Hidden Majority-Vote XOR Baseline

Architecture: `2`-spin XOR input -> all-to-all `2x2` hidden layer -> four replicated output spins. Prediction is the majority vote over the four output replicas.

- epochs: `500`
- workers: `32`
- Julia threads observed: `32`
- repeats per XOR case: `16`
- jobs per epoch: `32`
- free/nudged steps: `200` / `80`
- state mode: `discrete`
- dynamics: `metropolis`
- training rule: `analytic_teacher`
- optimizer: `Adam`
- learning rate: `0.05`
- beta: `0.5`
- final majority-vote MSE: `0.1162109375`
- final analog spin-score MSE: `0.1162109375`
- final accuracy: `1.0`
