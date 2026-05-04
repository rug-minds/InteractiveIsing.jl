struct LayerDisplayValue{F}
    f::F
end

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

displayable_hamiltonian_parameters(term, g) = ()

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
