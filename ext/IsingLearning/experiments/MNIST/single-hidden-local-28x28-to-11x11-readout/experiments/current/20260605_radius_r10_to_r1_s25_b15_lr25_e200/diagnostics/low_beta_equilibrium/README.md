# Low-Beta Equilibrium Diagnostics

These diagnostics test whether the r8 checkpoint can give useful low-beta behavior when we reduce stochastic readout noise instead of increasing the nudge strength.

Source checkpoint:

`../../r8/best_params.bin`

Scripts:

- `low_beta_multi_init_timeavg_readout.jl`
  - evaluates the r8 best checkpoint on a small balanced test subset;
  - repeats the same sample from multiple random initial states;
  - time-averages output replicas after a burn-in window;
  - also records low-beta nudged output response for beta values without training.
- `low_beta_output_trace_logger.jl`
  - uses a `ProcessAlgorithm` logger scheduled during relaxation;
  - writes output class scores versus sweep for a few samples;
  - plots whether the output has settled or keeps wandering.

The goal is diagnostic, not final accuracy. If low beta only works after averaging, the next training run should average equilibrium states in the gradient estimator rather than reading one best/final state.
