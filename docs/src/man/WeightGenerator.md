# Weight Generators

Weight generators define the couplings used when an adjacency matrix is built
from layers. They are not used during Monte Carlo updates; they are construction
tools for creating `J`.

The public interface is:

- `@WG` for most local, nearest-neighbor style generators.
- `WeightGenerator(func, NN, rng; symmetric=true)` when `NN`, `rng`, or other
  settings are runtime values.
- `AllToAllWeightGenerator(func, rng)` for dense inter-layer or in-layer
  generators.

## Local Generators

Use `@WG` with a keyword-style function:

```julia
wg = @WG (; dr) -> dr == 1 ? 1f0 : 0f0 NN = 1
```

This builds nearest-neighbor Ising couplings: edges at distance `dr == 1` get
weight `1f0`; all other candidate edges get weight `0f0`.

The generator function may use any subset of these keyword arguments:

- `dr`: Euclidean distance between the two candidate sites.
- `c1`: coordinate of the source site.
- `c2`: coordinate of the destination site.
- `dc`: integer coordinate displacement from `c1` to `c2`.

Examples:

```julia
nearest = @WG (; dr) -> dr == 1 ? 1.0 : 0.0 NN = 1

shell = @WG (; dr) -> dr <= sqrt(2) ? exp(-dr) : 0.0 NN = 1

anisotropic = @WG (; dc) -> dc[1] == 1 && dc[2] == 0 ? 1.0 : 0.25 NN = 1
```

Named functions are also fine:

```julia
function isingweights(; dr)
    return dr == 1 ? 1f0 : 0f0
end

wg = @WG isingweights NN = 1
```

## Neighborhood Size

`NN` controls which candidate edges are inspected.

```julia
wg1 = @WG (; dr) -> dr == 1 ? 1.0 : 0.0 NN = 1
wg2 = @WG (; dr) -> exp(-dr) NN = (2, 2, 1)
```

For a `D`-dimensional layer:

- `NN = n` checks offsets from `-n:n` in every dimension.
- `NN = (n1, n2, ...)` sets a different neighborhood radius per dimension.

The weight function may still return `0` for candidates inside this window. Zero
and `NaN` weights are skipped.

## Symmetric Couplings

Local `WeightGenerator`s are symmetric by default:

```julia
wg = @WG (; dr) -> randn() NN = 1
```

For same-layer couplings, this means each undirected edge is sampled once and
then mirrored, so `J_ij == J_ji`. This is the right default for energy-based
Ising Hamiltonians.

If you intentionally need directed candidate weights, opt out explicitly:

```julia
directed = @WG (; dr) -> randn() NN = 1 symmetric = false
```

Use `symmetric=false` only when the resulting adjacency is not meant to define a
standard symmetric energy term, or when you plan to handle symmetry yourself.

## Runtime Values

Macro keyword values such as `NN` are intended to be literals in ordinary use.
If `NN` or the RNG is decided at runtime, use the constructor:

```julia
function random_weight(; dr, c1=nothing, c2=nothing, dc=nothing)
    return dr == 1 ? randn() : 0.0
end

nn = 2
rng = Random.MersenneTwister(1)
wg = WeightGenerator(random_weight, nn, rng; symmetric = true)
```

For reproducible random weights, close over or pass an RNG:

```julia
rng = Random.MersenneTwister(1)
wg = @WG (; dr) -> dr == 1 ? randn(rng) : 0.0 NN = 1
```

The generator is evaluated when the graph adjacency is constructed. Reusing the
same `wg` after its RNG has advanced will produce a different random adjacency.

## Building Graphs

Use a weight generator as the layer weight generator:

```julia
wg = @WG (; dr) -> dr == 1 ? 1f0 : 0f0 NN = 1
layer = Layer(32, 32, StateSet(-1f0, 1f0), wg, Discrete(), Coords(0, 1, 0))
g = IsingGraph(layer; precision = Float32)
```

For multiple layers, put generators between layers:

```julia
inplane = @WG (; dr) -> dr == 1 ? 1f0 : 0f0 NN = 1
between = AllToAllWeightGenerator((; dr, c1, c2, dc) -> 0.01f0)

l1 = Layer(8, 8, StateSet(-1f0, 1f0), inplane, Discrete(), Coords(0, 1, 0))
l2 = Layer(8, 8, StateSet(-1f0, 1f0), inplane, Discrete(), Coords(0, 2, 0))

g = IsingGraph(l1, between, l2; precision = Float32)
```

Inter-layer generators are mirrored by graph construction, so the adjacency has
both directions for each inter-layer edge.

## All-To-All Generators

Use `AllToAllWeightGenerator` when every pair should be considered:

```julia
rng = Random.MersenneTwister(1)
dense = AllToAllWeightGenerator((; dr, c1, c2, dc) -> 0.1f0 * randn(rng))
```

This is useful for dense RBM-style layers or dense inter-layer couplings. For
local lattices, prefer `@WG` with a finite `NN`, because it avoids checking every
pair of spins.

## Practical Notes

- Prefer keyword-style generator functions: `(; dr, c1, c2, dc) -> ...`.
- Return `0` for candidate edges that should not exist.
- Return a concrete numeric type compatible with the graph precision.
- Same-layer random local generators should normally keep `symmetric=true`.
- The generated adjacency is fixed after graph construction; changing the weight
  generator does not mutate an existing graph.
