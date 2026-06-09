# Locally Connected 6x6 Window MNIST Replication

This folder is for reproducing the MNIST locally connected layer (LCL) findings
from arXiv:2601.21945 separately from the existing r8 local-readout rescue
diagnostics.

## Target

Replicate the layered-LCL MNIST experiment from [arXiv:2601.21945](https://arxiv.org/abs/2601.21945) as closely as possible:

- input: MNIST `28x28`
- first trainable stage: locally connected layer with a `6x6` receptive window and stride `1`
- no convolutional weight sharing
- output: `10` class nodes
- training protocol target from the paper: full MNIST, mini-batch `200`, `200` epochs, learning rate `1e-4`, `Minit=1`
- learning: EqProp-style contrastive training, with tangent nudging tracked as a first-class option

## Terminology

An LCL, or locally connected layer, is convolution-like local connectivity
without shared filters. Each output unit sees a local input patch, but its
weights are independent from every other output unit's patch weights.

## Run Rules

- Keep all paper-replication runs under `experiments/current/<series-name>` in
  this folder.
- Do not resume from unrelated checkpoints.
- Start with short diagnostics that verify response propagation and runtime
  before broad learning grids.
- Save settings, metrics, plots, and checkpoints for every learning run.
- Use descriptive run names; do not use vague `paper_*` labels.

## Initial Plan

1. Build a minimal LCL graph constructor with a `6x6` input-to-hidden window.
2. Verify shape, number of trainable input weights, field-input behavior, manager runtime, and checkpoint writing on a tiny batch.
3. Run a short learning diagnostic before full replication.
4. Only after one short run shows sane timing and non-degenerate learning, launch the full `200`-epoch/full-MNIST replication.

## Current Implementation Notes

- Script: `mnist_lcl_6x6_window_adam.jl`.
- Current hidden grid: valid-window `23x23`, because `(28 - 6) / 1 + 1 = 23`.
- Trainable input weights: `23 * 23 * 6 * 6 = 19044`.
- The structural input layer is not sampled. MNIST pixels are projected through the masked local input weights into a worker-local magnetic field.
- The sampled graph currently contains the hidden layer and output layer only. Hidden-to-output couplings are dense and bidirectional.
- Hidden intralayer couplings are absent in the first target, matching the paper's no-intralayer `6x6` LCL comparison before `+SQ`/`+4NSQ` variants.
- This reproduces the topology and manager training path in the Ising/Langevin codebase; it is not an exact XY-angle simulator.
