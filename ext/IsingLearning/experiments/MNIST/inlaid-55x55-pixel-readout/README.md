# Inlaid 55x55 Pixel Readout

Architecture: fixed `28x28` MNIST pixels are inlaid into a `55x55` layer with
live separator sites, then read out to `40` output spins.

- `schematic.png`: generated architecture schematic.
- `mnist_inlaid_input_training.jl`: manager-backed inlaid-input trainer.
- `experiments/current`: retained successful training/evaluation runs.
- `aggregate_plots`: comparison plots for this architecture family.

