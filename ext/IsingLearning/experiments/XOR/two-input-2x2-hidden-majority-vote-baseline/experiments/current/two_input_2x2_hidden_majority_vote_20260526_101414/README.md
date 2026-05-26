# Two-Input 2x2 Hidden Majority-Vote XOR Baseline

Architecture: `2`-spin XOR input -> all-to-all `2x2` hidden layer -> four replicated output spins. Prediction is the majority vote over the four output replicas.

- epochs: `5000`
- workers: `32`
- Julia threads observed: `32`
- repeats per XOR case: `32`
- jobs per epoch: `32`
- free/nudged steps: `200` / `200`
- dynamics: `BlockLangevin`, block size `4`
- optimizer: `Adam`
- learning rate: `0.002`
- beta: `1.0`
- final majority-vote MSE: `0.900390625`
- final analog spin-score MSE: `0.8724434672878807`
- final accuracy: `0.5`
