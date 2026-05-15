# Scalar 2 -> 8x8 -> 1 XOR With Local Hidden Couplings

This note documents the larger scalar-output LocalLangevin XOR experiment:

```text
2 physical input spins -> 8x8 hidden layer -> 1 scalar output spin
```

The file is:

```text
ext/IsingLearning/experiments/simple_langevin_xor/hidden8x8_2_64_1.jl
```

## Setup

The input is the honest two-bit bipolar XOR input:

```text
(0, 0) -> [-1, -1]
(0, 1) -> [-1, +1]
(1, 0) -> [+1, -1]
(1, 1) -> [+1, +1]
```

The target is one scalar bipolar output:

```text
XOR false -> -1
XOR true  -> +1
```

The architecture is:

- input-to-hidden: all-to-all random signed couplings;
- hidden layer: `8x8` continuous spins with `NN=1` local internal couplings;
- hidden-to-output: all-to-one random signed couplings;
- Hamiltonian terms: `Bilinear + MagField + masked Clamping`;
- no polynomial local potential;
- no double well;
- no zero initialization;
- no hand-designed XOR/corner-detector initialization.

The learning path is the normal IsingLearning path:

```text
Forward_and_Nudged -> contrastive_gradient -> Optimisers.update
```

The sampler is unadjusted `LocalLangevin`.

## Working Recipe

The scalar `2 -> 4 -> 1` result transferred to the larger hidden layer almost
directly. The important regime is still:

```text
T        = 0.005
stepsize = 0.4
β        = 2.0
Minit    = 8
```

The best tested run used:

```text
free/nudged = 600/600
lr          = 0.003
weight_decay = 0
input-hidden scale = 0.06
hidden-local scale = 0.04
hidden-output scale = 0.06
bias scale = 0.02
```

Result:

```text
run: ext/IsingLearning/experiments/simple_langevin_xor/runs/hidden8x8_600_600_T005_eta04_lr003
epoch 2000: MSE = 0.054399, accuracy = 1.0
mean outputs = [-0.694, 0.797, 0.776, -0.821]
```

The same recipe was rerun after adding graph saving:

```text
run: ext/IsingLearning/experiments/simple_langevin_xor/runs/hidden8x8_600_600_T005_eta04_lr003_savedgraph
epoch 2000: MSE = 0.067561, accuracy = 1.0
mean outputs = [-0.705, 0.822, 0.719, -0.730]
saved graph: hidden8x8_2_64_1_best_graph.jld2
```

The small difference between `0.054` and `0.068` is expected stochastic
variation from the relaxation/evaluation path. The important point is that the
same recipe solved the task twice.

Two nearby controls also worked:

```text
600/600, lr=0.002 -> MSE = 0.146542, accuracy = 1.0
1000/1000, lr=0.002 -> MSE = 0.083739, accuracy = 1.0
```

## What Made This Work

### 1. Same Dynamical Regime As The Small Scalar Run

The larger hidden layer did not need a fundamentally different learning rule.
It needed the same high-motion LocalLangevin regime:

```text
T = 0.005
stepsize = 0.4
```

Colder or more conservative Langevin runs previously tended to produce weak
sign solutions or get stuck. This setting gives the continuous hidden layer
enough motion to explore useful hidden configurations while still preserving a
learnable response to clamping.

### 2. Moderate Clamping

`β = 2` stayed useful. The result does not come from extreme clamping. Stronger
clamping is not automatically better because the useful EP signal is a
susceptibility-like response, not simply the largest forced displacement.

### 3. Local Hidden Couplings Did Not Prevent Learning

The hidden layer has `NN=1` internal couplings. This means the successful result
is not just a fully factorized hidden layer. The hidden state can locally
coordinate while the input and output are still globally connected to it.

### 4. Learning Rate Needed Slightly More Push Than The 2 -> 4 -> 1 Run

For this larger graph, `lr=0.003` did better than `lr=0.002` in the tested
2000-epoch window. The higher LR did not destabilize the run and reached the
best logged MSE.

### 5. Longer Relaxation Was Not Strictly Better

`1000/1000` also solved the task, but it was slower and did not beat the
`600/600, lr=0.003` run. This matches the smaller scalar experiment: more
relaxation is not monotonic. The useful signal depends on the dynamical regime,
not only proximity to a static endpoint.

## Interpretation

This result is important because it shows the scalar-output XOR recipe scales
from four hidden spins to a structured `8x8` hidden layer with local
interactions. The all-to-all input and output couplings still give the model
enough global access to learn XOR, while the hidden layer itself has local
Ising-style structure.

The remaining hard part is not scalar XOR capacity anymore. The current hard
part is transferring this reliability to more spatially constrained input and
output schemes, where the model cannot use global all-to-all access as freely.

## Hidden Local Coupling Distance

The hidden layer uses a local random weight generator. The `NN` setting controls
how far those hidden-hidden couplings reach inside the `8x8` layer:

- `NN = 0`: no hidden-hidden local couplings;
- `NN = 1`: only nearest-neighbor hidden-hidden couplings;
- `NN = 2`: hidden-hidden couplings out to distance two;
- `NN = 3`: hidden-hidden couplings out to distance three.

The first comparison kept every other setting fixed:

```text
T = 0.005
stepsize = 0.4
β = 2.0
lr = 0.003
free/nudged = 600/600
Minit = 8
hidden local weight scale = 0.04
```

| hidden `NN` | final MSE  | final accuracy | interpretation                                                             |
|------------:|-----------:|---------------:|:---------------------------------------------------------------------------|
|         `0` | `0.180805` |          `1.0` | no hidden-hidden local couplings; signs learned, but margins were weaker   |
|         `1` | `0.074700` |          `1.0` | best in this comparison                                                    |
|         `2` | `0.148370` |          `1.0` | learned, but weaker than `NN = 1`                                          |
|         `3` | `0.515669` |          `1.0` | signs were correct, but output magnitude stayed poor                       |

Changing `NN` also changes how many local hidden-hidden edges are present. With
the same per-edge scale, larger `NN` increases the total typical local field on
each hidden spin. To check whether `NN = 2` and `NN = 3` were worse only because
their local fields were larger, I also ran two variants with smaller
hidden-hidden weight scales:

| hidden `NN` | local scale | final MSE  | final accuracy | interpretation                                                        |
|------------:|------------:|-----------:|---------------:|:----------------------------------------------------------------------|
|         `2` |     `0.025` | `0.308226` |          `1.0` | worse than `NN = 2` with scale `0.04`                                  |
|         `3` |     `0.015` | `0.335248` |          `1.0` | better than `NN = 3` with scale `0.04`, still worse than `NN = 1`       |

Takeaway: for this all-to-all input/output scalar XOR task, one shell of local
hidden couplings is useful, but wider hidden-local coupling did not improve the
solution under this recipe. A plausible reason is that larger `NN` makes the
hidden layer more internally correlated, which can reduce the independent
hidden degrees of freedom available to separate the four XOR cases. This is an
interpretation of this sweep, not a proof.
