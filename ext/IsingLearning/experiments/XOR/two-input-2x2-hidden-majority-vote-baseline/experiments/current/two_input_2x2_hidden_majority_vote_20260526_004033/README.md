# Two-Input 2x2 Hidden Majority-Vote XOR Baseline

Architecture: `2`-spin XOR input -> all-to-all `2x2` hidden layer -> four replicated output spins. Prediction is the sign of the mean output, equivalent to a majority vote over the four output replicas.

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
- final majority-score MSE: `0.8388097625473878`
- final accuracy: `1.0`
