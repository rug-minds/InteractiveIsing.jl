# MNIST 784-120-40 Baseline Performance Update Timing

- purpose: one fresh full-training-epoch timing run after the baseline performance/design edits
- start policy: wait for old baseline process `1068836` before launching
- workers: `32`
- epochs: `1`
- batch size: `128`
- train per class: `5421`
- test per class: `1`
- train eval per class: `0`
- sweeps: `500`
- beta: `5.0`
- learning rate: `0.0015`
- weight decay: `0.0`
- comparison target: old full baseline training epochs were about `368-373 s`; the active old-code revisit is about `445-449 s`
- measured field to compare: `seconds` in `mnist_784_120_40_adam.csv`
