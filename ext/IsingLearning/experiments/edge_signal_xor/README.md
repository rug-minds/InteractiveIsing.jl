# Edge Signal XOR Experiment

This folder contains one experiment file:

- `edge_signal_xor.jl`

The graph is:

```text
2 input spins -> left edge of 8x8 hidden layer -> right edge -> 1 output spin
```

The hidden layer has local symmetric couplings controlled by `EDGE_XOR_HIDDEN_NN`.
The input and output connections are edge-only, not all-to-all.

Run a quick learning search without response traces:

```julia
ENV["EDGE_XOR_SKIP_RESPONSE"] = "true"
ENV["EDGE_XOR_EPOCHS"] = "500"
ENV["EDGE_XOR_NNS"] = "0,1,2"
include("ext/IsingLearning/experiments/edge_signal_xor/edge_signal_xor.jl")
```

Run learning plus response traces:

```julia
ENV["EDGE_XOR_SKIP_RESPONSE"] = "false"
ENV["EDGE_XOR_RESPONSE_PRE_SWEEPS"] = "60"
ENV["EDGE_XOR_RESPONSE_SWEEPS"] = "80"
include("ext/IsingLearning/experiments/edge_signal_xor/edge_signal_xor.jl")
```

The response trace starts from a relaxed source XOR input, switches to a target
XOR input, and logs after every full sweep. It records output response, left and
right edge response, total hidden response, and the average response column.

The smoke test verified:

- The adjacency is symmetric.
- Each input spin connects to 8 hidden spins on the left edge.
- The output spin connects to 8 hidden spins on the right edge.
- Hidden local couplings are generated through `@WG`.

First checked run:

- folder: `runs/edge_signal_xor_20260512_151913`
- `NN=1`, 500 epochs, `Minit=4`, free/nudged `400/400`
- best logged MSE was `0.856821` at epoch `200`, accuracy `0.75`

That run did not solve XOR. It is useful as a baseline because it confirms the
edge-only graph trains through the existing IsingLearning path and produces the
response trace files, but the recipe still needs tuning before comparing learned
versus random signal propagation.

Second checked run:

- folder: `runs/edge_signal_xor_20260512_153558`
- `NN=5`, 1000 epochs, `Minit=4`, free/nudged `400/400`
- hidden layer was made non-periodic for this run, which is the correct boundary
  condition for an edge-propagation experiment
- best logged MSE was `0.860759` at epoch `200`, accuracy `0.75`
- final MSE was `1.022974`, final accuracy `0.5`

This exact `NN=5` recipe did not work. It initially had some class separation,
then lost it. That suggests wider local hidden coupling alone is not enough; the
next useful probes are temperature fraction, input/output edge coupling scale,
and relaxation length.

Targeted grid from `runs/edge_signal_grid_20260512_154608`:

| hidden NN | T fraction | edge scale | hidden scale | best MSE | best accuracy | best epoch |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.015 | 0.12 | 0.04 | 0.650530 | 1.0 | 400 |
| 1 | 0.025 | 0.16 | 0.04 | 0.604403 | 1.0 | 300 |
| 1 | 0.050 | 0.16 | 0.04 | 0.614388 | 1.0 | 200 |
| 2 | 0.015 | 0.16 | 0.025 | 0.680686 | 1.0 | 300 |
| 2 | 0.025 | 0.16 | 0.025 | 0.529323 | 1.0 | 400 |
| 2 | 0.050 | 0.20 | 0.025 | 0.561450 | 1.0 | 200 |
| 3 | 0.025 | 0.20 | 0.015 | 0.586935 | 1.0 | 300 |
| 3 | 0.050 | 0.20 | 0.015 | 0.609628 | 1.0 | 400 |
| 5 | 0.025 | 0.20 | 0.008 | 0.760501 | 1.0 | 200 |
| 5 | 0.050 | 0.20 | 0.008 | 0.801339 | 1.0 | 100 |
| 5 | 0.025 | 0.25 | 0.005 | 0.649038 | 1.0 | 1 |
| 5 | 0.080 | 0.25 | 0.005 | 0.711917 | 1.0 | 1 |

The best tested point is `NN=2`, temperature fraction `0.025`, edge scale
`0.16`, hidden-local scale `0.025`. It reached `1.0` accuracy but not low MSE.
The output signs are learned, but the output magnitudes remain small. The next
step should test whether this is caused by weak output fields, finite
temperature, or insufficient relaxation by logging raw output samples and
effective output fields.

Longer validation check:

- folder: `runs/edge_signal_xor_20260512_160432`
- same best grid region: `NN=2`, T fraction `0.025`, edge scale `0.16`,
  hidden-local scale `0.025`
- training free/nudged stayed `500/500`
- validation relaxation was increased to `3000`
- result did not improve: final MSE `1.036243`, final accuracy `0.25`

This means the previous MSE around `0.53` was not simply because validation was
too short. Longer relaxation from random validation starts can settle into
different attractors or erase the weak sign separation. The next diagnostic
should compare validation from fixed warm starts versus random starts, and log
the raw output samples rather than only their mean.

Single-relaxation snapshot diagnostic:

- folder: `runs/relaxation_snapshots_20260512_162128`
- graph: best graph from `runs/edge_signal_grid_20260512_154608/05_NN2_T0p025_io0p16_h0p025_lr0p002/best_graph.jld2`
- input case: first XOR case
- dynamics: unadjusted `LocalLangevin`, stepsize `0.4`
- saved snapshots every `10` full sweeps up to `500` full sweeps

The state was not frozen after 500 full sweeps. Consecutive saved snapshots in
the late part of the run, from 400 to 500 full sweeps, had:

| interval | value |
|---|---:|
| average cosine similarity | `0.956254` |
| minimum cosine similarity | `0.942140` |
| maximum cosine similarity | `0.970813` |
| average relative change norm | `0.294919` |
| relative change norm range | `0.246095` to `0.340104` |

The output spin also kept moving. Over the whole 500-sweep trajectory it ranged
from `-0.982753` to `-0.333840`, and the final output was `-0.839982`.

So the issue is not just a too-short 500-sweep validation run. With this
finite-temperature Langevin setup the trajectory keeps fluctuating around the
low-energy region instead of becoming nearly static. If we want a deterministic
energy-minimization diagnostic, we need a separate low-noise or zero-noise
validation mode rather than interpreting finite-temperature Langevin samples as
frozen states.
