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
  the default scalar container for that term.
- A concrete vector, matrix, or scalar container is checked against the graph
  and converted where the term asks for graph precision.
- A function `g -> ...` is evaluated during instantiation.
- `NoEnsure(x)` skips automatic resizing/conversion but still runs checks.
- `Force(x)` skips automatic resizing/conversion and hard checks.

The default `Ising()` shortcut expands to:

- `Quadratic(...)`
- `Bilinear(...)`
- `MagField(...)`

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

`MagField(; b = nothing, c = nothing)`

Represents ``-c \sum_i b_i s_i``. The default field is a zero `ConstFill`
with graph-state length. `b = 1` means a uniform field of one in graph
precision.

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

`CoulombHamiltonian(...)`

Uses the same parameter/internal split, with `scaling` as a symbolic parameter
and FFT buffers, screening lengths, lattice constants, and scratch arrays in
internal storage.

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
