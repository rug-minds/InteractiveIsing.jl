# Edge Signal Time-Averaged Readout

This folder tests a different readout for the learned edge-signal XOR graph:
instead of classifying from one final output spin, it averages the output spin
after a burn-in period.

The averaging is implemented by `OutputAverager`, an experiment-local
`ProcessAlgorithm`. It is scheduled together with the Langevin dynamics, so the
measurement is still done through the StatefulAlgorithms machinery:

```text
burn-in dynamics
then repeat dynamics, recording one output sample after each full sweep
```

For `average_sweeps = 50` and `sample_every_sweeps = 1`, this means exactly 50
output samples total.

## What This Tests

Time averaging can reduce readout noise when the temperature is a little too
high for the current coupling scale. It does not fix a graph whose energy
landscape does not contain the correct attractors.

If the graph is a good equilibrium classifier, then after enough burn-in the
time-averaged output should stay near the correct sign and have useful
magnitude. If it only works for a very early burn-in window and then degrades,
that is a transient response, not a stable Ising attractor.

## First Probe

Run folder:

```text
runs/time_average_20260512_171109
```

This loaded:

```text
../edge_signal_xor/runs/edge_signal_grid_20260512_154608/05_NN2_T0p025_io0p16_h0p025_lr0p002/best_graph.jld2
```

The useful setting in that quick probe was:

```text
temperature factor: 0.5 or 0.25
burn-in: 2 full sweeps
average: 50 samples, one after each full sweep
```

Observed examples:

| T factor | burn-in | average samples | MSE | accuracy | means |
|---:|---:|---:|---:|---:|---|
| `0.5` | `2` | `50` | `0.275640` | `1.0` | `[-0.798, 0.013, 0.773, -0.813]` |
| `0.25` | `2` | `50` | `0.284277` | `1.0` | `[-0.793, 0.008, 0.772, -0.760]` |

This is not a clean solution. One class still has mean output close to zero,
so the margin is weak. More importantly, longer burn-in made the result worse.

## Attractor Check

A follow-up zero-temperature check was run and stopped once the pattern was
clear. At `T = 0`, using the same learned graph:

| burn-in | average samples | MSE | accuracy | means |
|---:|---:|---:|---:|---|
| `5` | `50` | `0.817509` | `0.5` | `[-0.012, -0.021, -0.085, -0.732]` |
| `25` | `50` | `1.339728` | `0.25` | `[0.024, -0.030, -0.790, -0.783]` |
| `100` | `50` | `0.785217` | `0.5` | `[-0.037, -0.054, -0.029, -0.793]` |
| `250` | `50` | `1.042343` | `0.5` | `[0.720, 0.723, -0.053, -0.842]` |

This means the current learned edge graph is not a robust attractor-based XOR
classifier. The early two-sweep readout is a transient response. It should not
be counted as solving the equilibrium Ising classification task.

## Current Conclusion

The time-averaged readout is still useful, but only after the graph has learned
the right attractors. For this edge-signal graph, the next fix should target the
learning setup and architecture, not the readout:

- train with a validation objective that checks late low-temperature attractors;
- lower or schedule temperature during learning as weights grow;
- use stronger or better normalized output coupling;
- keep using time averaging for readout noise, but reject solutions that fail
  the zero-temperature or very-low-temperature attractor check.
