# IsingGraphs

`IsingGraph` is the main container used by the package. It holds

- the spin state,
- the adjacency / coupling structure,
- the Hamiltonian,
- layer metadata,
- and runtime extras such as the active process list.

This page documents the current constructor API. Older examples that pass `type = Continuous` are stale. The current constructors use positional `Continuous()` / `Discrete()` arguments instead.

## Two constructor styles

There are two user-facing ways to build a graph:

```julia
IsingGraph(nx::Int, args...; periodic = true, precision = Float32, adj = nothing, fastwrite = false)
```

Use this for a single 2D or 3D lattice.

```julia
IsingGraph(layer1::IsingLayerData, items...; precision = Float32, adj = nothing, index_set = nothing, callback! = identity)
```

Use this when you want multiple explicit layers inside one graph.

## Start from the 3D example

The current `examples/3D Graph.jl` file is a good template:

```julia
using InteractiveIsing
using InteractiveIsing.Processes

function isingweights(; dr)
    dr == 1 ? 1f0 : 0f0
end

wg = @WG isingweights NN = 1

g = IsingGraph(
    100,
    100,
    10,
    Continuous(),
    wg,
    LatticeConstants(1f0, 1f0, 1f0),
    StateSet(-1.5f0, 1.5f0),
    Ising(c = ConstVal(0f0), b = StateLike(ConstFill, 0f0)) +
    CoulombHamiltonian(recalc = 2000);
    periodic = (:x, :y),
)

createProcess(g)
```

This call builds one 3D layer with shape `(100, 100, 10)`. The arguments mean:

- `100, 100, 10`: the lattice size.
- `Continuous()`: the state type for that layer.
- `wg`: the in-layer weight generator used to build the adjacency.
- `LatticeConstants(...)`: physical spacing used by distance-based generators and 3D electrostatic terms.
- `StateSet(-1.5f0, 1.5f0)`: the allowed state interval for a continuous layer.
- `Ising(...) + CoulombHamiltonian(...)`: the Hamiltonian attached to the graph.
- `periodic = (:x, :y)`: periodic boundaries in `x` and `y`, open in `z`.

The key distinction is that `IsingGraph(100, 100, 10, ...)` creates one 3D lattice, not ten separate 2D layers.

## Single-layer graphs

The single-layer constructor is the most compact entry point:

```julia
wg2d = @WG (; dr) -> dr == 1 ? 1f0 : 0f0 NN = 1

g2d = IsingGraph(
    64,
    64,
    Continuous(),
    wg2d,
    StateSet(-1.5f0, 1.5f0),
    Ising() + Quartic();
    periodic = (:x, :y),
)
```

```julia
wg3d = @WG (; dr) -> exp(-dr) NN = (1, 1, 1)

g3d = IsingGraph(
    64,
    64,
    8,
    Continuous(),
    wg3d,
    LatticeConstants(1.0, 1.0, 2.0),
    StateSet(-1.5f0, 1.5f0),
    Ising();
    periodic = (:x, :y),
)
```

After the dimensions, the constructor commonly receives:

- a state type such as `Continuous()` or `Discrete()`,
- a weight generator created with `@WG`,
- optional `LatticeConstants(...)`,
- optional `StateSet(...)`,
- and an optional Hamiltonian.

If no Hamiltonian is given, the default is `Ising()`. If no weight generator is given, the graph is still created, but the off-diagonal couplings remain zero unless you provide `adj` yourself.

The constructor initializes the spin state from the layer state set, builds the adjacency, reconstructs the Hamiltonian against the final graph shape, and stores the result in `g.hamiltonian`.

## Single 3D layer vs explicit multi-layer graph

These are different modeling choices:

- `IsingGraph(nx, ny, nz, ...)` means one 3D lattice with one topology and one in-layer weight generator.
- `IsingGraph(layer1, wg12, layer2, wg23, layer3, ...)` means several layers that share one graph state vector but keep separate layer identities.

Use the single 3D constructor when the material is naturally one 3D grid. Use the multi-layer form when each layer should stay individually addressable or when couplings between layers should be specified explicitly.

## Multi-layer graphs

For explicit layer stacks, first build the layers with `Layer(...)`:

```julia
inplane = @WG (; dr) -> dr == 1 ? 1f0 : 0f0 NN = 1
between = @WG (; dr) -> 0.25f0 NN = 1

l1 = Layer(
    64,
    64,
    Continuous(),
    inplane,
    StateSet(-1.5f0, 1.5f0),
    Coords(y = 0, x = 0, z = 0),
)

l2 = Layer(
    64,
    64,
    Continuous(),
    inplane,
    StateSet(-1.5f0, 1.5f0),
    Coords(y = 0, x = 0, z = 1),
)

g = IsingGraph(
    l1,
    between,
    l2,
    Ising() + Clamping();
    precision = Float32,
)
```

This constructor reads the arguments from left to right:

- every `Layer(...)` adds one layer,
- every weight generator placed between two layers couples exactly those neighboring layers,
- the Hamiltonian applies to the full graph after all layers have been assembled.

Two practical rules matter here:

- Between-layer generators need layer coordinates. Give each layer a `Coords(...)` value if you want inter-layer couplings.
- `Coords` are easiest to read in keyword form: `Coords(y = ..., x = ..., z = ...)`.

You can extend the pattern to more than two layers:

```julia
g = IsingGraph(layer1, wg12, layer2, wg23, layer3, hamiltonian)
```

If you omit `wg12`, `wg23`, and so on, the layers remain uncoupled.

## Layer construction

`Layer(args...)` is a thin wrapper around the layer parser. In practice the important pieces are:

```julia
Layer(
    dims...,
    Continuous() | Discrete(),
    weight_generator,
    StateSet(...),
    LatticeConstants(...),
    Coords(...),
)
```

The parser distinguishes these objects by type, so after the dimension arguments the exact order is flexible. The examples above keep a consistent order because it is easier to read.

For a continuous layer, `StateSet(a, b)` defines the interval `[a, b]`. For a discrete layer, `StateSet(v1, v2, v3, ...)` enumerates the allowed values.

## Inspecting the result

Useful accessors after construction:

```julia
state(g)          # full graph state
adj(g)            # graph adjacency
g[1]              # first layer view
size(g[1])        # layer shape
graphidxs(g[1])   # indices of that layer in the full state vector
g.hamiltonian     # reconstructed Hamiltonian terms
```

The layer views returned by `g[i]` are the normal entry point for per-layer interaction after the graph has been created.

The companion page [`Hamiltonians`](Hamiltonians.md) shows how to choose and compose the energy terms passed into the constructor.
