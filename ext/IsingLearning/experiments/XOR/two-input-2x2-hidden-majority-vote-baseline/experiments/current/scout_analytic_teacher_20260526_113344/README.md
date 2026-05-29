# Two-Input 2x2 Hidden Majority-Vote XOR Baseline

Architecture: `2`-spin XOR input -> all-to-all `2x2` hidden layer -> four replicated output spins. Prediction is the majority vote over the four output replicas.

- epochs: `300`
- workers: `32`
- Julia threads observed: `32`
- repeats per XOR case: `16`
- jobs per epoch: `32`
- free/nudged steps: `80` / `80`
- dynamics: `BlockLangevin`, block size `8`
- training rule: `analytic_teacher`
- optimizer: `Adam`
- learning rate: `0.05`
- beta: `0.5`
- final majority-vote MSE: `0.666015625`
- final analog spin-score MSE: `0.676565804610815`
- final accuracy: `1.0`
