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

## Layer Terms

Use `LayerTerm` for Hamiltonians whose geometry, buffers, or topology belong to
one layer rather than to the full graph. A layer term stores only an integer
selector before graph construction:

```julia
struct SomeLayerTerm{P,I} <: LayerTerm
    layer::Int
    parameters::P
    internal::I
end
```

The constructor should accept `layer = 1` or another explicit integer keyword.
It must not store a graph or layer object:

```julia
function SomeLayerTerm(; layer = 1, c = nothing)
    params = Parameters(parameter(; c, type = AbstractArray,
                                  default = ConstVal(1f0),
                                  ensure = ensure_isinggraph_scalar))
    internal = InternalPlan((;)) do plan, layer_model
        return SomeInternal(size(layer_model), topology(layer_model))
    end
    return SomeLayerTerm(Int(layer), params, internal)
end
```

During `instantiate(term, graph)`, the template system resolves
`graph[term.layer]` and instantiates parameters and internals against that
layer. The layer provides the graph-like methods needed by template defaults:
`statelen(layer)`, `graphstate(layer)`, and `adj(layer)`.

Layer terms normally implement `_calculate`, not public `calculate`:

```julia
@inline function _calculate(::ΔH, term::SomeLayerTerm, layer, proposal)
    return term.c[] * local_energy_change(layer, at_idx(proposal), to_val(proposal))
end

@inline function _calculate(::d_iH, term::SomeLayerTerm, layer, local_idx)
    return term.c[] * local_derivative(layer, local_idx)
end
```

The generic public `calculate` method for `LayerTerm` first checks whether the
proposal or spin index belongs to `term.layer`. Out-of-layer calls return zero.
In-layer calls translate the global graph index to a layer-local index before
calling `_calculate`.

Mutable layer terms should implement `_update!`:

```julia
@inline function _update!(::Metropolis, term::SomeLayerTerm, layer, proposal)
    update_layer_cache!(term, layer, at_idx(proposal), delta(proposal))
end
```

The public `update!` wrapper ignores rejected proposals and out-of-layer
proposals before calling `_update!`.

If a layer term needs completely custom semantics, it may overload public
`calculate` directly. That bypasses the standard scope rejection and index
translation, so it should be used deliberately.

## Wrapping Ordinary Terms On A Layer

Use `ToLayer(layer, hamiltonian)` when the implementation is already an
ordinary Hamiltonian but you want to evaluate it on one layer as a mini graph.
This is separate from native `LayerTerm` implementations:

- `LayerTerm` is for terms whose implementation is intrinsically layer-shaped.
- `ToLayer` reuses an ordinary `HamiltonianTerm` or `HamiltonianTerms` bundle
  on `graph[layer]`.

```julia
h = ToLayer(2, ExtField(b = 1))
```

`ToLayer` stores only the integer layer selector and the wrapped Hamiltonian.
During graph construction, it instantiates the wrapped Hamiltonian against the
bound layer, so parameters are compact layer-sized values rather than full
graph-sized masked values.

The wrapper itself implements only `_calculate` and `_update!` hooks:

```julia
_calculate(f, wrapper, layer, args...) =
    calculate(f, inner(wrapper), layer, args...)
```

Public `calculate` and `update!` still come from the generic `LayerTerm`
wrapper, so out-of-layer proposals return zero, rejected or out-of-layer
updates are ignored, and global spin indices are translated to layer-local
indices before the wrapped Hamiltonian sees them.

Wrapped terms must be able to run against the layer graph-like interface:
`eltype(layer)`, `statelen(layer)`, `graphstate(layer)`, and `adj(layer)`.
Terms such as `ExtField`, `Bilinear`, `PolynomialHamiltonian`, and `Clamping`
use that interface directly. Terms that require full-graph-specific APIs should
remain global or become native `LayerTerm`s.

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
