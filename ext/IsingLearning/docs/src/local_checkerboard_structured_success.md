# Local Checkerboard Structured Success

This note records the first local checkerboard XOR setup in this branch that
reaches both required diagnostics:

```text
accuracy = 1.0
best repeated-state MSE < 0.1
```

The successful run is:

```text
ext/IsingLearning/experiments/local_checkerboard_xor/runs/structured_and_seed_20260511_052236/
```

The saved graph is:

```text
ext/IsingLearning/experiments/local_checkerboard_xor/runs/structured_and_seed_20260511_052236/structured_2x2_T0.001_ib2.0_f3.5_o0.8_ob0.4_b0.1_lr0.002/structured_2x2_T0.001_ib2.0_f3.5_o0.8_ob0.4_b0.1_lr0.002_best_graph.jld2
```

## Result

Best logged validation result:

```text
MSE = 0.095703125
accuracy = 1.0
decision scores = [-2.0, 1.0, 1.625, -1.375]
```

The run starts from the structured local circuit and then runs the normal
StatefulAlgorithms/contrastive-gradient path for 300 epochs. The best saved graph is the
best validation checkpoint, not necessarily the final epoch.

## Exact Recipe

```text
input/output/hidden size = 2x2 -> 2x2 -> 2x2
input protocol = checkerboard A/B active sites frozen to +1
inactive input bits = not frozen
input default = fixed b = -2.0 on input-code sites
hidden seed = explicit weights/biases for A, B, AND feature route
output = two local checkerboard readout channels
state = discrete
init = all -1 background
dynamics = Metropolis
temperature = 0.001
free relaxation = 700
nudged relaxation = 150
Minit = 4
eval repeats = 32
beta = 0.1
lr = 0.002
weight decay = 0
same-layer internal scale = 0
structured feature scale = 3.5
structured output scale = 0.8
structured output bias = 0.4
```

Run command used for the short training confirmation:

```bash
ISING_STRUCTURED_CONFIGS='structured_2x2_T0.001_ib2.0_f3.5_o0.8_ob0.4_b0.1_lr0.002' \
ISING_STRUCTURED_INIT=minus \
ISING_STRUCTURED_INTERNAL_SCALE=0.0 \
ISING_STRUCTURED_EPOCHS=300 \
ISING_STRUCTURED_LOG_EVERY=50 \
ISING_STRUCTURED_MINIT=4 \
ISING_STRUCTURED_EVAL_REPEATS=32 \
ISING_STRUCTURED_FREE_RELAXATION=700 \
ISING_STRUCTURED_NUDGED_RELAXATION=150 \
ISING_STRUCTURED_THREADS=8 \
ISING_STRUCTURED_TEMPS=0.001 \
ISING_STRUCTURED_INPUT_BIASES=2.0 \
ISING_STRUCTURED_FEATURE_SCALES=3.5 \
ISING_STRUCTURED_OUTPUT_SCALES=0.8 \
ISING_STRUCTURED_OUTPUT_BIASES=0.4 \
julia --project=ext/IsingLearning \
  ext/IsingLearning/experiments/local_checkerboard_xor/structured_and_seed_search.jl
```

## What Changed

The earlier scalar-readout random local checkerboard searches had two coupled
problems:

- absent input bits were physically undefined, because `0` meant "no clamped
  spins" rather than a relaxed `-1` state;
- a single scalar checkerboard readout had to both discover and amplify the XOR
  separation.

The successful setup fixes those separately:

- inactive input bits are still not clamped, but a fixed negative input field
  gives them a physical default;
- the hidden layer receives explicit `A`, `B`, and `AND` feature weights;
- the output is supervised through two local readout channels, one per output
  checkerboard mask.

This does not prove random local graphs discover XOR from scratch. It shows the
local checkerboard system can represent and stably run XOR when the architecture
contains the needed `AND` pathway.

## What The A/B/AND Seed Means

This is not an initial hidden state. The hidden spins are still initialized by
the normal experiment setting (`init = all -1 background` in this run), and the
free/nudged phases relax from those states. The word "seed" means that the
initial graph parameters are not purely random: `apply_structured_and_seed!` in
`local_checkerboard_stabilized_search.jl` writes a small symmetric local circuit
into the adjacency matrix and magnetic-field vector before training starts.

For the successful `2x2 -> 2x2 -> 2x2` run, the code chooses three hidden
spins from the hidden-layer checkerboard masks:

```text
hA   = first hidden checkerboard-A site
hAND = second hidden checkerboard-A site
hB   = first hidden checkerboard-B site
```

Then it writes these symmetric input-to-hidden weights, with
`f = structured_feature_scale = 3.5`:

```text
for i in input_A_mask:
    J[i, hA]   = J[hA, i]   = f / length(input_A_mask)
    J[i, hAND] = J[hAND, i] = f / length(input_A_mask)

for i in input_B_mask:
    J[i, hB]   = J[hB, i]   = f / length(input_B_mask)
    J[i, hAND] = J[hAND, i] = f / length(input_B_mask)
```

Because the Ising bilinear term is `E_J = -1/2 sum_ij J_ij s_i s_j`, positive
`J` makes connected spins prefer the same sign. So `hA` is aligned with the
active A checkerboard mask, `hB` is aligned with the active B mask, and `hAND`
receives excitation from both masks.

The AND-like hidden spin also gets a negative field:

```text
b[hAND] -= f
```

The magnetic-field term is `E_b = -sum_i b_i s_i`, so negative `b[hAND]`
favours `hAND = -1`. That means one active input mask is not enough to robustly
turn `hAND` on; both A and B inputs are needed to overcome this negative bias.
That is the hand-coded local AND pathway.

Finally, the hidden features are wired to the two output checkerboard readout
masks. With `o = structured_output_scale = 0.8`, the true-output mask gets:

```text
J[hA, output_true]   = +o
J[hB, output_true]   = +o
J[hAND, output_true] = -2o
```

and the false-output mask gets the opposite:

```text
J[hA, output_false]   = -o
J[hB, output_false]   = -o
J[hAND, output_false] = +2o
```

This implements the XOR sign structure `A + B - 2AB`: cases `(1,0)` and
`(0,1)` drive the true readout, while `(0,0)` and `(1,1)` drive the false
readout. A small output bias is also added:

```text
b[output_false] += 0.4
b[output_true]  -= 0.4
```

That bias favours the false channel as the default, which is useful because
`(0,0)` freezes no input spins.
