# Hamiltonians

Hamiltonians are built from small term objects and combined with `+`.
The graph constructor instantiates those terms against the graph, so the user
can write compact graph-independent input while each term receives graph-sized
data internally.

```julia
h = Ising() + Quartic() + Sextic()
```

```julia
g = IsingGraph(
    64,
    64,
    Continuous(),
    wg,
    StateSet(-1.5f0, 1.5f0),
    h;
    periodic = (:x, :y),
)
```

## Front-End Use

Most users only need the constructors. Constructor inputs follow the same
rules across template-based terms:

- `nothing` means use the term default.
- A plain scalar such as `1` is converted to the graph precision and wrapped in
  the explicit-input container for that term.
- A singleton vector such as `[1]` is treated like an explicit scalar and
  expanded to graph length.
- A concrete vector, matrix, or scalar container is checked against the graph
  and converted where the term asks for graph precision.
- A function `g -> ...` is evaluated during instantiation.
- `NoEnsure(x)` skips automatic resizing/conversion but still runs checks.
- `Force(x)` skips automatic resizing/conversion and hard checks.

The default `Ising()` shortcut expands to:

- `Quadratic(...)`
- `Bilinear(...)`
- `ExtField(...)`

```julia
Ising(; c = nothing, b = nothing, adj = nothing, localpotential = nothing)
```

where:

- `c` is the on-site quadratic coupling,
- `b` is the magnetic-field vector,
- `adj` or `J` is the bilinear coupling matrix,
- `localpotential` is the graph-sized local polynomial potential.

Examples:

```julia
Ising()
```

```julia
Ising(c = ConstVal(0f0), b = 0)
```

```julia
Ising(b = UniformArray)
```

```julia
Ising(adj = g -> adj(g), localpotential = g -> adj(g).diag)
```

## Built-In Terms

`Bilinear(; adj = nothing, J = nothing)`

Represents ``-1/2 \sum_{ij} J_{ij}s_i s_j``. If neither `adj` nor `J` is
provided, it uses `adj(g)`. `J` must be an `n x n` matrix for `n` graph states.

`ExtField(; b = nothing, c = nothing)`

Represents ``-c \sum_i b_i s_i``. The default field is a zero `ConstFill`
with graph-state length. `b = 1` means a uniform field of one in graph
precision using mutable uniform storage.

`Quadratic`, `Quartic`, `Sextic`, `Octic`

Represent local polynomial terms ``c l_i s_i^n``. `c` is scalar-like and
`localpotential` is state-like. By default, `localpotential` is `adj(g).diag`.

```julia
Quartic(c = ConstVal(1f0))
```

`Clamping(β, y)`

Represents ``β/2 (s_i - y_i)^2``. The target `y` is instantiated as a mutable
state-length vector so it can be changed with `clamp!`.

`DepolField(...)`

Stores the coupling `c` as a symbolic parameter and keeps graph-derived state
such as boundary layers and accumulated depolarisation in internal storage.
This is a layer-bound term; pass `layer = i` to bind it to layer `i`.

`CoulombHamiltonian(...)`

Uses the same parameter/internal split, with `scaling` as a symbolic parameter
and FFT buffers, screening lengths, lattice constants, and scratch arrays in
internal storage. This is also layer-bound:

```julia
CoulombHamiltonian(layer = 2, recalc = 1000)
```

Layer-bound terms only contribute for proposals on their bound layer. Proposals
on other layers return zero for local energy changes and do not update the
term's internal caches.

## Wrapping Global Terms Onto A Layer

Use `ToLayer(layer, hamiltonian)` when an ordinary Hamiltonian should operate
on one layer as if that layer were a small graph:

```julia
h = ToLayer(2, ExtField(b = 1))
```

The wrapped term is instantiated against `graph[2]`, so its parameters are
layer-sized. Proposals outside layer 2 contribute zero.

You can also wrap a bundle:

```julia
h = ToLayer(2, Ising(c = ConstVal(0), b = 0.5))
```

This is useful when the usual term implementation already works on graph-like
accessors such as `graphstate`, `statelen`, and `adj`, but you want local
scope. For example, `Bilinear` inside `ToLayer` uses the layer-local adjacency
and ignores cross-layer edges.

`ToLayer` composes with global terms:

```julia
h =
    Bilinear() +
    ToLayer(2, ExtField(b = 1)) +
    ToLayer(3, Ising(c = ConstVal(0), b = 0.5))
```

Here the global `Bilinear()` still contributes everywhere using the full sparse
adjacency, while the wrapped terms contribute only on their bound layers.

## Global And Layer-Bound Terms

Hamiltonians can mix global terms and layer-bound terms:

```julia
h =
    Ising(c = ConstVal(0f0), b = 0) +
    CoulombHamiltonian(layer = 2, recalc = 1000) +
    DepolField(layer = 2)
```

`Bilinear` remains global by default and is still represented by one sparse
matrix over the full graph. This is usually the fastest representation for
interactions, including disconnected or block-structured graphs.

`ExtField` is also global. To work with one layer's field values, use layer
accessors or views of the graph-sized field parameter rather than making
`ExtField` layer-bound. True `LayerTerm`s are intended for terms whose internal
geometry or cached state cannot be represented as a simple full-graph vector or
matrix.

If you want an ordinary `ExtField` to be scoped to one layer instead, wrap it:

```julia
ToLayer(2, ExtField(b = 1))
```

## Containers

For scalar and state-like parameters, prefer the custom containers when the
parameter is spatially uniform or should be cheap to update. See
[Hamiltonian Containers](@ref) for details.

Common examples:

```julia
Ising(c = ConstVal(0f0))
Ising(b = ConstFill(0f0))
Ising(b = UniformArray)
Ising(b = OffsetArray)
```

`StateLike(...)` is legacy. New terms should express graph-sized defaults with
ordinary defaults plus ensure functions, and front-end users should pass normal
values, containers, or graph functions.

## Developer Interface

New Hamiltonian terms can either be written freestyle by overloading
`calculate`, or they can opt into the template convention:

```julia
struct MyTerm{P,I} <: HamiltonianTerm
    parameters::P
    internal::I
end
```

Use `parameters` only for symbolic Hamiltonian parameters. Use `internal` for
buffers, plans, caches, graph constants, and other implementation state. See
[Hamiltonian Term Templates](@ref) for the backend pattern.
