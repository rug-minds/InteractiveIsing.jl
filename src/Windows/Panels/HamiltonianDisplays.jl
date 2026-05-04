"""
    LayerDisplayValue(f)

Wrapper for a layer-local display function used by the Windows Hamiltonian
viewer. The function is called as `f(layer)` and must return an array shaped
like that layer, or another shape supported by the panel's layer renderer.

Use `layer_display` instead of constructing this wrapper directly in
ordinary extension code.
"""
struct LayerDisplayValue{F}
    f::F
end

"""
    HamiltonianDisplaySpec(name, value; source = :parameter, origin = :owned,
                           info = "", colormap = :viridis,
                           colorrange = :layer)

Description of one selectable Hamiltonian visualization in the Windows
interface.

`value` can be a graph-sized vector, which is split by the selected graph
layer, or a `LayerDisplayValue`, which is evaluated for the selected
layer. `colorrange = :layer` uses the layer state range; `colorrange = :data`
uses the displayed data range.
"""
struct HamiltonianDisplaySpec{V}
    name::Symbol
    source::Symbol
    origin::Symbol
    info::String
    value::V
    colormap::Symbol
    colorrange::Symbol
end

const COULOMB_CHARGES_NAME = Symbol("ρ")

"""
    HamiltonianDisplaySpec(name::Symbol, value; kwargs...)

Construct a Hamiltonian display specification. Prefer `parameter_display`
for ordinary graph-sized Hamiltonian parameters and `layer_display` for
derived layer-shaped data.
"""
function HamiltonianDisplaySpec(
    name::Symbol,
    value;
    source = :parameter,
    origin = :owned,
    info = "",
    colormap = :viridis,
    colorrange = :layer,
)
    return HamiltonianDisplaySpec(
        name,
        Symbol(source),
        Symbol(origin),
        string(info),
        value,
        Symbol(colormap),
        Symbol(colorrange),
    )
end

"""
    layer_display(name::Symbol, f; kwargs...) -> HamiltonianDisplaySpec

Create a display spec for a derived quantity. `f` is called as `f(layer)` when
the user selects the display entry, so it can compute values from the currently
selected layer.

# Example

```julia
InteractiveIsing.Windows.hamiltonian_visualizations(term::MyTerm, g) = [
    layer_display(:local_energy, layer -> my_local_energy(term, layer);
        colormap = :viridis,
        colorrange = :data,
    ),
]
```
"""
function layer_display(
    name::Symbol,
    f;
    source = :derived,
    origin = :internal,
    info = "",
    colormap = :viridis,
    colorrange = :data,
)
    return HamiltonianDisplaySpec(
        name,
        LayerDisplayValue(f);
        source,
        origin,
        info,
        colormap,
        colorrange,
    )
end

"""
    displayable_hamiltonian_parameters(term, g)

Return the Hamiltonian parameter names that should be offered by the Windows
Hamiltonian viewer.

The default implementation returns all parameters whose instantiated values are
graph-sized vectors, so most parameter-template Hamiltonians need no custom UI
code. Override this method when a Hamiltonian should hide a compatible
parameter or enforce a specific order:

```julia
InteractiveIsing.Windows.displayable_hamiltonian_parameters(term::MyTerm, g) =
    (:field, :bias)
```

For fully custom or derived displays, override `hamiltonian_visualizations`
instead.
"""
displayable_hamiltonian_parameters(term, g) = _state_sized_parameter_names(term, g)

"""
    hamiltonian_visualizations(term, g) -> Vector{HamiltonianDisplaySpec}

Return the display entries shown for `term` in the Windows Hamiltonian panel.

The fallback turns `displayable_hamiltonian_parameters` into ordinary
parameter displays. Custom Hamiltonians can either rely on that fallback,
override `displayable_hamiltonian_parameters`, or return bespoke specs with
`parameter_display` and `layer_display`.
"""
function hamiltonian_visualizations(term, g)
    return _parameter_visualizations(term, g, displayable_hamiltonian_parameters(term, g))
end

hamiltonian_visualizations(term::PolynomialHamiltonian, g) = _parameter_visualizations(term, g, (:lp,))
hamiltonian_visualizations(term::MagField, g) = _parameter_visualizations(term, g, (:b,))
hamiltonian_visualizations(term::Clamping, g) = _parameter_visualizations(term, g, (:y,))

function _parameter_visualizations(term, g, names)
    specs = HamiltonianDisplaySpec[]
    for name in names
        spec = parameter_display(term, name, g)
        isnothing(spec) || push!(specs, spec)
    end
    return specs
end

function _state_sized_parameter_names(term, g)
    params = parameters(term)
    isnothing(params) && return ()

    raw_entries = getfield(params, :entries)
    names = Symbol[]
    for name in propertynames(raw_entries)
        raw_entry = getproperty(raw_entries, name)
        entry_value = raw_entry isa Parameter ? value(raw_entry) : getproperty(params, name)
        _is_state_sized(entry_value, g) && push!(names, name)
    end
    return Tuple(names)
end

"""
    parameter_display(term, name::Symbol, g; kwargs...) -> Union{HamiltonianDisplaySpec, Nothing}

Build a display spec for a graph-sized Hamiltonian parameter. Returns `nothing`
when `name` does not exist or when the instantiated parameter value is not
state-sized.

This is the lowest-friction extension point for ordinary Hamiltonian fields:

```julia
InteractiveIsing.Windows.hamiltonian_visualizations(term::MyTerm, g) = [
    parameter_display(term, :field, g; colormap = :thermal),
]
```
"""
function parameter_display(
    term,
    name::Symbol,
    g;
    source = :parameter,
    colormap = :viridis,
    colorrange = :layer,
)
    params = parameters(term)
    isnothing(params) && return nothing

    raw_entries = getfield(params, :entries)
    haskey(raw_entries, name) || return nothing

    raw_entry = getproperty(raw_entries, name)
    entry_value = raw_entry isa Parameter ? value(raw_entry) : getproperty(params, name)
    _is_state_sized(entry_value, g) || return nothing

    info = getfield(params, :info)
    entry_origin = raw_entry isa Parameter ? Symbol(nameof(typeof(origin(raw_entry)))) : :spec
    entry_info = haskey(info, name) ? string(getproperty(info, name)) : ""

    return HamiltonianDisplaySpec(
        name,
        entry_value;
        source,
        origin = entry_origin,
        info = entry_info,
        colormap,
        colorrange,
    )
end

function hamiltonian_visualizations(term::CoulombHamiltonian, g)
    return HamiltonianDisplaySpec[
        layer_display(
            :u,
            _ -> copy(term.u);
            info = "Electrostatic potential u on charge planes",
            colormap = :viridis,
            colorrange = :data,
        ),
        layer_display(
            COULOMB_CHARGES_NAME,
            _ -> copy(term.ρ);
            info = "Real charge density on charge planes",
            colormap = :viridis,
            colorrange = :data,
        ),
    ]
end

_is_state_sized(val, g) = val isa AbstractArray && ndims(val) == 1 && length(val) == nstates(g)
