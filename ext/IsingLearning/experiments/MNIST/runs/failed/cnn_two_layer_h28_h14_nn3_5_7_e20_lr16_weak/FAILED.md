# Failed Run: H14 CNN-Style NN Sweep With LR16

Use of this file: explains why this two-hidden-layer CNN-style sweep is under `failed`.

This run tested `28^2 input fields -> 28^2 hidden1 -> 14^2 hidden2 -> 40 outputs` with local NN/fanout radii `3`, `5`, and `7`, 32 workers, batchsize `32`, trainable same-layer couplings, and the same 16x mean-normalized LR scale that worked for the `11x11` second hidden layer.

It was useful as a diagnostic, but it was not a good run:

- radius 3 best small-test accuracy: `0.325`;
- radius 5 best small-test accuracy: `0.42`;
- radius 7 stayed near chance, best `0.125`.

The follow-up with radius 5 and an 8x LR scale learned well, so this file is kept to document that the larger `14x14` second hidden layer needs a lower update scale than the `11x11` run.
