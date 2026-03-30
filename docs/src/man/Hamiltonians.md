# Hamiltonians

Hamiltonians in `InteractiveIsing.jl` are built from small term objects and combined with `+`. The graph constructor then reconstructs those terms against the final graph shape and stores the result in `g.hamiltonian`.

In other words, you usually write a compact symbolic description first:

```julia
h = Ising() + Quartic() + Clamping()
```

and `IsingGraph(...)` turns that into graph-sized arrays, adjacency-backed terms, and any other graph-dependent data structures that the terms need.

## Basic pattern

Pass the Hamiltonian as one of the positional arguments to `IsingGraph`:

```julia
wg = @WG (; dr) -> dr == 1 ? 1f0 : 0f0 NN = 1

g = IsingGraph(
    64,
    64,
    Continuous(),
    wg,
    StateSet(-1.5f0, 1.5f0),
    Ising() + Quartic();
    periodic = (:x, :y),
)
```

Terms are composed with `+`:

```julia
h = Ising() + Quartic() + Sextic()
```

The result is a `HamiltonianTerms` object containing each term.

## The `Ising()` shortcut

`Ising()` is the main convenience constructor. It expands to

- `Quadratic(...)`,
- `Bilinear(...)`,
- `MagField(...)`.

The keyword arguments let you override those pieces:

```julia
Ising(; c = nothing, b = nothing, adj = nothing, localpotential = nothing)
```

The common meanings are:

- `c`: coefficient used by the quadratic on-site term,
- `b`: magnetic field,
- `adj`: adjacency used by the bilinear interaction,
- `localpotential`: local potential used by the polynomial term.

If you do not override them, the defaults are graph-derived:

- `Bilinear` uses `adj(g)`,
- `Quadratic` uses `adj(g).diag` as its local potential,
- `MagField` starts from a zero field.

## Common term constructors

The most useful built-in terms for graph construction are:

- `Ising(...)`: quadratic + bilinear + magnetic field.
- `Quadratic(...)`, `Quartic(...)`, `Sextic(...)`, `Octic(...)`: on-site polynomial terms.
- `Clamping(β, y)`: penalty term `β/2 * (s_i - y_i)^2`.
- `CoulombHamiltonian(; recalc = N, scaling = 1f0, screening = Inf32)`: long-range electrostatic term for the 3D setup used in the example graphs.

Examples:

```julia
h_ising = Ising()
```

```julia
h_landau = Ising(c = ConstVal(0f0)) + Quartic(c = ConstVal(1f0)) + Sextic(c = ConstVal(1f0))
```

```julia
h_clamped = Ising() + Clamping(1f0)
```

```julia
h_coulomb = Ising(c = ConstVal(0f0), b = StateLike(ConstFill, 0f0)) +
            CoulombHamiltonian(recalc = 2000)
```

## Choosing parameter containers

Many term fields accept either concrete arrays or graph-dependent placeholders. The placeholders are useful because the graph size is not known until the graph has been constructed.

The most common helpers are:

- `ConstVal(x)`: a scalar-like constant. Good for coefficients such as `c`.
- `StateLike(ArrayType, default)`: build something with the same length as the graph state when the Hamiltonian is reconstructed.
- `g -> ...`: any function of the graph, for fully custom graph-derived fields.

Examples:

```julia
Ising(c = ConstVal(0f0))
```

```julia
Ising(b = StateLike(ConstFill, 0f0))
```

```julia
Bilinear(adj = g -> adj(g))
```

```julia
Quartic(localpotential = g -> adj(g).diag)
```

`StateLike` is particularly useful when the field should track the graph size automatically.

## Recipes

### Nearest-neighbor Ising in 2D

```julia
wg = @WG (; dr) -> dr == 1 ? 1f0 : 0f0 NN = 1
g = IsingGraph(64, 64, Continuous(), wg, Ising(); periodic = (:x, :y))
```

### Landau-style local energy

```julia
wg = @WG (; dr) -> dr == 1 ? 1f0 : 0f0 NN = 1
h = Ising(c = ConstVal(0f0)) + Quartic(c = ConstVal(1f0)) + Sextic(c = ConstVal(1f0))
g = IsingGraph(64, 64, Continuous(), wg, StateSet(-1.5f0, 1.5f0), h)
```

### Supervised / clamped setup

```julia
h = Ising() + Clamping(1f0)
```

`Clamping` reconstructs its target vector `y` to the full graph size, so it composes naturally with the normal graph constructor.

### 3D electrostatic example

```julia
h = Ising(c = ConstVal(0f0), b = StateLike(ConstFill, 0f0)) +
    CoulombHamiltonian(recalc = 2000)
```

This is the pattern used in the 3D example graph. At the moment `CoulombHamiltonian` is tied to the 3D layer geometry and is best understood as a specialized term for that setup.

## Inspecting the reconstructed Hamiltonian

After graph construction, the realized terms live in `g.hamiltonian`:

```julia
g.hamiltonian
gethamiltonian(g.hamiltonian, Bilinear)
gethamiltonian(g.hamiltonian, CoulombHamiltonian)
```

This is useful when you want to inspect the actual adjacency or parameter storage that a term ended up using.

The constructor overview on [`IsingGraphs`](IsingGraphs.md) shows how these Hamiltonians are attached to single-layer and multi-layer graphs.
