# Two-Hidden Local MNIST Pair Grid

- architecture family: `28^2 input fields -> 11^2 hidden1 -> 5^2 hidden2 -> 40 outputs`
- selected pairs: `7:3`
- workers / batchsize / epochs: `1` / `2` / `1`
- train/test per class: `1` / `1`
- optimizer: `adam`
- learning rates W0/W12/W2O/B: `0.012` / `0.012` / `0.012` / `0.0012`
- free/nudge sweeps: `75` / `75`
- beta: `5.0`
- gradient normalization: `mean`
