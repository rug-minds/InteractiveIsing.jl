# Two-Input 2x2 Hidden Majority-Vote XOR Baseline

Architecture: `2`-spin XOR input -> all-to-all `2x2` hidden layer -> four replicated output spins. Prediction is the majority vote over the four output replicas.

- epochs: `400`
- workers: `32`
- Julia threads observed: `32`
- repeats per XOR case: `32`
- jobs per epoch: `32`
- free/nudged steps: `80` / `80`
- dynamics: `BlockLangevin`, block size `8`
- training rule: `target_free`
- optimizer: `Adam`
- learning rate: `0.005`
- beta: `0.5`
- final majority-vote MSE: `1.45263671875`
- final analog spin-score MSE: `1.0737152961503975`
- final accuracy: `0.5`
