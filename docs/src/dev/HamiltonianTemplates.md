# Hamiltonian Term Templates

The only required Hamiltonian interface is still `calculate`. The template
layer is optional. It exists to make graph-instantiated Hamiltonian terms
regular: symbolic parameters go in one place, internal implementation state
goes in another, and the constructor automatically turns front-end input into
graph-sized data.

## Shape Of A Template Term

A template term usually looks like this:

```julia
struct SomeTerm{P,I} <: HamiltonianTerm
    parameters::P
    internal::I
end
```

Use `parameters` for values that appear in the symbolic Hamiltonian:
couplings, fields, local potentials, and user-facing model parameters.

Use `internal` for implementation details: scratch buffers, FFT plans,
precomputed constants, graph shape, caches, counters, and derived state that
does not belong in the symbolic expression.

Terms without internal state can use only:

```julia
struct SomeTerm{P} <: HamiltonianTerm
    parameters::P
end
```

## Parameters

Create symbolic parameters with `parameter`.

```julia
params = Parameters(
    parameter(;
        c,
        type = AbstractArray,
        default = ConstVal(1f0),
        default_type = UniformArray,
        ensure = ensure_isinggraph_scalar,
        info = "Coupling constant",
    ),
)
```

A `ParameterSpec` is pre-instantiation. A `Parameter` is post-instantiation.
`Parameters` stores the entries in a named tuple and stores `info` and `units`
separately. Access is forwarded through the term, so calculation code can write
`hterm.c` instead of `parameters(hterm).c`.

Constructor input rules:

- `nothing` uses the default.
- plain values go through `ensure` using `default_type` as the fill storage;
- singleton arrays such as `[1]` are expanded to graph-sized storage for
  state-like parameters;
- storage types such as `Vector` or `UniformArray` are filled through
  `filltype`;
- graph functions are evaluated by ensure functions that support them;
- `NoEnsure(x)` skips ensure but still checks;
- `Force(x)` skips ensure and hard checks.

`default` and `default_type` are deliberately separate:

- `default` is the omitted-input value.
- `default_type` is the storage used for explicit scalar/singleton input.
- if `default_type` is omitted, it is inferred from `typeof(default)`.

Useful ensure functions:

- `ensure_isinggraph_eltype`
- `ensure_isinggraph_scalar`
- `ensure_isinggraph_state_length`
- `ensure_isinggraph_state_vector`
- `ensure_isinggraph_adjacency`

Ensure functions can be composed with tuples:

```julia
ensure = (ensure_isinggraph_state_length, ensure_isinggraph_eltype)
```

Auto-filled containers must implement:

```julia
filltype(::Type{MyArray}, value, dims...) = MyArray(value, dims...)
```

The template does not fall back to dense `fill` for unknown storage types.
Missing `filltype` methods should fail early so container authors know which
interface they need to implement.

## Internal State

For simple graph-derived values, use `ArrayPlan` or `TypePlan`.

```julia
internal = TypePlan(g -> some_graph_value(g))
```

For larger implementations, use one `InternalPlan` and write the initialization
as a normal block:

```julia
internal = InternalPlan((; option)) do plan, g
    config = plan.values
    T = eltype(g)

    return SomeInternal(
        zeros(T, statelen(g)),
        config.option,
    )
end
```

The result should usually be a typed subtype of `InternalImplementation`.
That keeps runtime fields concrete after instantiation.

```julia
struct SomeInternal{T,V} <: InternalImplementation
    buffer::Vector{T}
    data::V
end
```

## Instantiation

For ordinary template terms, the generic instantiation is enough:

```julia
instantiate(parameters(term), g)
instantiate(internal(term), g)
```

The generic term method reconstructs the term with instantiated parameters and
internal state.

If final parameters depend on instantiated internals, write a custom
`instantiate` for that term and reuse the same pieces:

```julia
function instantiate(term::SomeTerm, g::AbstractIsingGraph)
    params = instantiate(parameters(term), g)
    internals = instantiate(internal(term), g)

    final_parameters = # term-specific adjustment

    return SomeTerm(final_parameters, internals)
end
```

Keep this custom method narrow. It should only contain the dependency that the
generic path cannot express.

## Calculation Code

Calculation methods should read parameters and internals through the term:

```julia
@inline function calculate(::ΔH, hterm::SomeTerm, model, proposal)
    return hterm.c[] * hterm.field[at_idx(proposal)]
end
```

`getproperty` first checks real fields, then parameter names, then internal
fields. This keeps calculation code short without custom accessor functions.

## Front-End Constructors

Expose a small constructor with keyword arguments. Keep user-facing names
stable and map them to parameter names inside the constructor.

```julia
function SomeTerm(; c = nothing, field = nothing)
    params = Parameters(
        parameter(; c, type = AbstractArray, default = ConstVal(1f0),
                  ensure = ensure_isinggraph_scalar),
        parameter(; field, type = AbstractArray, default = ConstFill(0),
                  default_type = UniformArray,
                  ensure = (ensure_isinggraph_state_length, ensure_isinggraph_eltype)),
    )
    return SomeTerm(params)
end
```

Do not put buffers or implementation-only options in `parameters`. Put those in
`internal` or in the `InternalPlan` values.
