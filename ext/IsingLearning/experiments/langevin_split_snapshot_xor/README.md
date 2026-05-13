# Langevin Split-Snapshot XOR Experiment

This folder contains a local experiment for testing a split-snapshot EqProp
variant with `LocalLangevin`.

The free phase runs in two parts:

1. Run a short early relaxation and copy `free_dynamics.model` into the shared
   `equilibrium_state` context field.
2. Continue the same free dynamics to a later endpoint.

The plus/minus nudged phases restore from the early `equilibrium_state`, but
the contrastive gradient still uses the later free endpoint graph:

```julia
contrastive_gradient(free_endpoint, plus_state, minus_state, beta)
```

This is intentionally implemented with `@Routine`, `@CompositeAlgorithm`,
`@repeat`, and normal worker `Process` objects. There is no manual `step!`
loop in this file.

Current implementation notes:

- Sampler: direct `LocalLangevin(adjusted=false)`.
- No scheduler wrapper is used here; the earlier scheduler wrapper hit a
  Processes context merge failure in this route.
- Input is explicit bipolar checkerboard input from the included local
  checkerboard experiment code.
- Hamiltonian has bilinear couplings, trainable magnetic field, and output
  pattern clamping. Polynomial/double-well local potentials are not used.
- Nudged branches restart from the early snapshot through the shared
  `equilibrium_state` field, matching the existing stable checkerboard EqProp
  composite shape.

Short smoke result on May 11, 2026:

- The composite constructs and trains end to end.
- A 60-epoch short run over two local-Langevin settings did not solve XOR.
  Best observed result was MSE `0.874838`, accuracy `0.75`.
- That means the process wiring is fixed, but this split-snapshot Langevin
  recipe still needs better dynamics/temperature/relaxation tuning.
