# MNIST CNN-Style Two-Layer NN Grid

Use of this folder: compare local square NN/fanout radii for the two-hidden-layer CNN-style MNIST architecture.

- architecture family: `28^2 input fields -> 28^2 hidden1 -> H2^2 hidden2 -> 40 outputs`
- hidden2 sides: `14`
- local NN/fanout radii: `3,5,7`
- workers / batchsize / epochs: `32` / `32` / `20`
- train/test per class: `100` / `20`
- gradient normalization: `mean`

| Rank | Config | H2 Side | NN Radius | Best Test Accuracy | Best Epoch | Final Test Accuracy |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `cnn_h1_28_h2_14_r5` | 14 | 5 | 0.42 | 16 | 0.25 |
| 2 | `cnn_h1_28_h2_14_r3` | 14 | 3 | 0.325 | 17 | 0.3 |
| 3 | `cnn_h1_28_h2_14_r7` | 14 | 7 | 0.125 | 0 | 0.1 |

Plot: `cnn_two_layer_nn_summary.png`
Summary: `cnn_two_layer_nn_summary.csv`
