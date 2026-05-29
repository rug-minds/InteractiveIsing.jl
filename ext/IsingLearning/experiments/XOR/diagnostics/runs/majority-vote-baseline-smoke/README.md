# Two-Input 2x2 Hidden Majority-Vote XOR Baseline

Architecture: `2`-spin XOR input -> all-to-all `2x2` hidden layer -> four replicated output spins. Prediction is the majority vote over the four output replicas.

- epochs: `1`
- workers: `1`
- Julia threads observed: `1`
- repeats per XOR case: `1`
- jobs per epoch: `4`
- free/nudged steps: `1` / `1`
- dynamics: `BlockLangevin`, block size `4`
- optimizer: `Adam`
- learning rate: `0.002`
- beta: `1.0`
- final majority-vote MSE: `0.625`
- final analog spin-score MSE: `0.8989526830566031`
- final accuracy: `0.5`
