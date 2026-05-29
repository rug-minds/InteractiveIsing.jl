# Two-Input 2x2 Hidden Majority-Vote XOR Baseline

Architecture: `2`-spin XOR input -> all-to-all `2x2` hidden layer -> four replicated output spins. Prediction is the majority vote over the four output replicas.

- epochs: `1`
- workers: `1`
- Julia threads observed: `32`
- repeats per XOR case: `1`
- jobs per epoch: `4`
- free/nudged steps: `4` / `4`
- dynamics: `BlockLangevin`, block size `8`
- training rule: `ep`
- optimizer: `Adam`
- learning rate: `0.005`
- beta: `0.05`
- final majority-vote MSE: `0.5625`
- final analog spin-score MSE: `0.8586505497735896`
- final accuracy: `1.0`
