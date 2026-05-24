# Inlaid Input MNIST

Use of this folder: saved run artifacts for the 55x55 inlaid-input MNIST architecture.

- architecture: `55x55 partially dynamic input -> 40 output replicas`
- fixed pixels/live separators: `784` / `2241`
- workers/batchsize: `32` / `256`
- train/test per class: `100` / `40`
- free/nudged/eval sweeps: `75` / `75` / `100`
- reads free/nudged/eval: `1` / `1` / `1`
- optimizer: `adam`
- lr/decay/min: `0.01` / `0.995` / `0.001`
- beta: `20.0`
- parameter/applied bias clip: `2.0` / `20.0`
- output replica/competition couplings: `0.1` / `0.5`
- train live separator readout: `false`
- best test accuracy: `0.61` at epoch `30`
- final test accuracy: `0.61`

## Post-hoc best checkpoint evaluation

- reads=1, sweeps=100: accuracy=0.5875, loss=6.56, prediction counts=135;39;44;39;34;28;22;24;20;15
- reads=3, sweeps=100: accuracy=0.61, loss=6.46, prediction counts=132;39;45;38;34;30;22;28;19;13
- reads=5, sweeps=150: accuracy=0.61, loss=6.4, prediction counts=133;38;45;38;34;33;19;27;19;14
