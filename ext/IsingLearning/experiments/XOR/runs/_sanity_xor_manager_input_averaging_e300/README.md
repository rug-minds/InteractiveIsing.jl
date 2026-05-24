# XOR Manager Input Averaging

Clean `2 -> 4 -> 2x4` XOR demonstrator.

The worker algorithm is `AveragedContrastiveStep(LayerContrastiveStep(layer))`.
Each manager job runs several repeats inside one already-started worker execution,
then `FlushAtEnd()` merges all worker-local contrastive buffers once.

- epochs: `300`
- workers: `32`
- repeats per XOR case: `128`
- chunks per XOR case: `8`
- jobs per epoch: `32`
- free/nudged steps: `300` / `300`
- beta: `2.0`
- optimizer: `adam`
- optimizer learning rate: `0.002`
- temperature: `0.005`
- LocalLangevin stepsize: `0.4`
