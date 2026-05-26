# Two-Input 2x2 Hidden Majority-Vote XOR Baseline

Architecture: `2`-spin XOR input -> all-to-all `2x2` hidden layer -> four replicated output spins. Prediction is the majority vote over the four output replicas.

- epochs: `1`
- workers: `1`
- Julia threads observed: `32`
- repeats per XOR case: `1`
- jobs per epoch: `4`
- free/nudged steps: `4` / `4`
- dynamics: `BlockLangevin`, block size `8`
- training rule: `target_free`
- target-free sign: `free_minus_target`
- optimizer: `Adam`
- learning rate: `0.005`
- beta: `0.5`
- final majority-vote MSE: `0.875`
- final analog spin-score MSE: `1.0261742816671058`
- final accuracy: `0.75`
