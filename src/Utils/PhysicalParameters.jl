export PhysicalScales,
       PhysicalUnitSpec,
       MissingPhysicalScale,
       MissingPhysicalRole,
       physicalunits,
       physicalscales,
       setphysicalscale!,
       internalvalue,
       physicalvalue,
       setphysical!

"""
    PhysicalScales(; energy, length, temperature, charge, dipole, state)

Mutable reference scales used to convert Unitful inputs into internal numeric
Hamiltonian values. Missing scales are allowed until a Unitful value needs
them, at which point conversion throws a targeted error.
"""
struct PhysicalScales{E,L,Temp,Q,D,S}
    energy::E
    length::L
    temperature::Temp
    charge::Q
    dipole::D
    state::S
end

function PhysicalScales(;
    energy = nothing,
    length = nothing,
    temperature = nothing,
    charge = nothing,
    dipole = nothing,
    state = nothing,
)
    return PhysicalScales(
        Ref{Any}(energy),
        Ref{Any}(length),
        Ref{Any}(temperature),
        Ref{Any}(charge),
        Ref{Any}(dipole),
        _state_scale_refs(state),
    )
end

"""
    PhysicalUnitSpec(; energy = 0, length = 0, temperature = 0, charge = 0,
                     dipole = 0, state = 0, state_layer = nothing, role = :custom)

Describe the physical dimensions of one internal Hamiltonian parameter in
terms of the graph's reference scales.
"""
struct PhysicalUnitSpec{P,L,R}
    powers::P
    state_layer::L
    role::R
end

function PhysicalUnitSpec(;
    energy::Integer = 0,
    length::Integer = 0,
    temperature::Integer = 0,
    charge::Integer = 0,
    dipole::Integer = 0,
    state::Integer = 0,
    state_layer = nothing,
    role::Symbol = :custom,
)
    powers = (;
        energy = Int(energy),
        length = Int(length),
        temperature = Int(temperature),
        charge = Int(charge),
        dipole = Int(dipole),
        state = Int(state),
    )
    return PhysicalUnitSpec(powers, state_layer, role)
end

physicalunits(; kwargs...) = PhysicalUnitSpec(; kwargs...)

struct MissingPhysicalScale{P,R,V} <: Exception
    parameter::P
    role::R
    value::V
end

struct MissingPhysicalRole{P,V} <: Exception
    parameter::P
    value::V
end

function Base.showerror(io::IO, err::MissingPhysicalScale)
    print(
        io,
        "Missing physical scale `",
        err.role,
        "` while converting Hamiltonian parameter `",
        err.parameter,
        "` from ",
        typeof(err.value),
        ". Add it with `PhysicalScales(",
        err.role,
        " = 1u\"...\")` or `setphysicalscale!(physicalscales(g), :",
        err.role,
        ", 1u\"...\")`.",
    )
end

function Base.showerror(io::IO, err::MissingPhysicalRole)
    print(
        io,
        "Missing physical unit metadata for Hamiltonian parameter `",
        err.parameter,
        "` with Unitful input ",
        typeof(err.value),
        ". Add `units = physicalunits(...)` to that parameter spec or pass a plain internal numeric value.",
    )
end

_state_scale_refs(::Nothing) = Dict{Any,Base.RefValue{Any}}()
_state_scale_refs(state::AbstractDict) = Dict{Any,Base.RefValue{Any}}(k => Ref{Any}(v) for (k, v) in state)
_state_scale_refs(state::NamedTuple) = Dict{Any,Base.RefValue{Any}}(Symbol(k) => Ref{Any}(v) for (k, v) in pairs(state))
_state_scale_refs(state) = Dict{Any,Base.RefValue{Any}}(:default => Ref{Any}(state))

_isunitful(value) = value isa Unitful.AbstractQuantity
_contains_unitful(value) = _isunitful(value)
_contains_unitful(value::AbstractArray) = any(_contains_unitful, value)
_contains_unitful(value::Tuple) = any(_contains_unitful, value)
_contains_unitful(value::NamedTuple) = any(_contains_unitful, values(value))
_contains_unitful(value::Base.RefValue) = _contains_unitful(value[])
_contains_unitful(::Nothing) = false

"""
    physicalscales(model)

Return the physical scale context attached to a graph or layer. If no explicit
context exists yet, a missing-scale context is created on the graph.
"""
function physicalscales(model::M) where {M<:AbstractIsingGraph}
    return get!(addons(model), :physical_scales, PhysicalScales())
end

function physicalscales(layer::L) where {L<:AbstractIsingLayer}
    return physicalscales(graph(layer))
end

physicalscales(scales::PhysicalScales) = scales
physicalscales(scales::NamedTuple) = _physical_scales_argument(scales)
physicalscales(::Nothing) = PhysicalScales()

"""
    _temperature_unit_spec(value, scales)

Choose the conversion metadata for a temperature input. Explicit temperature
scales accept physical temperature units; otherwise energy-compatible Unitful
values are treated as `k_B T` and converted with the graph energy scale.
"""
function _temperature_unit_spec(value::T, scales::S) where {T,S}
    scale_context = physicalscales(scales)
    if value isa Unitful.AbstractQuantity &&
        isnothing(scale_context.temperature[]) &&
        !isnothing(scale_context.energy[])

        try
            Unitful.uconvert(Unitful.NoUnits, value / scale_context.energy[])
            return physicalunits(energy = 1, role = :temperature_energy)
        catch
            # Non-energy Unitful temperatures keep the ordinary temperature
            # role so missing-scale errors report the correct extension point.
        end
    end
    return physicalunits(temperature = 1, role = :temperature)
end

function _physical_scales_argument(scales::PhysicalScales)
    return scales
end

function _physical_scales_argument(scales::NamedTuple)
    return PhysicalScales(; scales...)
end

function _physical_scales_argument(::Nothing)
    return PhysicalScales()
end

"""
    setphysicalscale!(scales, role, value; layer = nothing)

Update a mutable physical scale reference and return `scales`.
"""
function setphysicalscale!(scales::PhysicalScales, role::Symbol, value; layer = nothing)
    if role === :state
        key = isnothing(layer) ? :default : layer
        state_refs = getfield(scales, :state)
        state_refs[key] = Ref{Any}(value)
        return scales
    end

    role in (:energy, :length, :temperature, :charge, :dipole) ||
        throw(ArgumentError("Unknown physical scale role `$(role)`."))
    getfield(scales, role)[] = value
    return scales
end

function _scale_ref(scales::PhysicalScales, role::Symbol)
    return getfield(scales, role)
end

function _state_scale(scales::PhysicalScales, model, state_layer)
    state_refs = getfield(scales, :state)
    keys_to_try = Any[]
    !isnothing(state_layer) && push!(keys_to_try, state_layer)
    if model isa AbstractIsingLayer
        push!(keys_to_try, layeridx(model))
    end
    push!(keys_to_try, :default)

    for key in keys_to_try
        haskey(state_refs, key) && return state_refs[key][]
    end
    return nothing
end

function _scale_power(scale, power::Integer)
    power == 0 && return 1
    if power > 0
        result = scale
        for _ in 2:power
            result *= scale
        end
        return result
    end

    result = inv(scale)
    for _ in 2:(-power)
        result *= inv(scale)
    end
    return result
end

function _scale_value(scales::PhysicalScales, spec::PhysicalUnitSpec, role::Symbol, power::Integer, parameter::Symbol, value)
    power == 0 && return 1
    scale = _scale_ref(scales, role)[]
    isnothing(scale) && throw(MissingPhysicalScale(parameter, role, value))
    return _scale_power(scale, power)
end

function _scale_value(scales::PhysicalScales, spec::PhysicalUnitSpec, ::Val{:state}, power::Integer, parameter::Symbol, value, model)
    power == 0 && return 1
    scale = _state_scale(scales, model, spec.state_layer)
    isnothing(scale) && throw(MissingPhysicalScale(parameter, :state, value))
    return _scale_power(scale, power)
end

function _scale_product(scales::PhysicalScales, spec::PhysicalUnitSpec, model, parameter::Symbol, value)
    p = spec.powers
    product = 1
    product *= _scale_value(scales, spec, :energy, p.energy, parameter, value)
    product *= _scale_value(scales, spec, :length, p.length, parameter, value)
    product *= _scale_value(scales, spec, :temperature, p.temperature, parameter, value)
    product *= _scale_value(scales, spec, :charge, p.charge, parameter, value)
    product *= _scale_value(scales, spec, :dipole, p.dipole, parameter, value)
    product *= _scale_value(scales, spec, Val(:state), p.state, parameter, value, model)
    return product
end

"""
    internalvalue(value, spec, scales, model; parameter = :value)

Convert a Unitful value into the internal dimensionless value implied by
`spec` and `scales`. Plain numeric values are returned unchanged.
"""
function internalvalue(value, spec::Nothing, scales, model; parameter::Symbol = :value)
    _contains_unitful(value) && throw(MissingPhysicalRole(parameter, value))
    return value
end

function internalvalue(value, spec::PhysicalUnitSpec, scales, model; parameter::Symbol = :value)
    _contains_unitful(value) || return value
    scale_context = physicalscales(scales)
    return _internalvalue(value, spec, scale_context, model, parameter)
end

function _internalvalue(value::Unitful.AbstractQuantity, spec::PhysicalUnitSpec, scales::PhysicalScales, model, parameter::Symbol)
    scale = _scale_product(scales, spec, model, parameter, value)
    return Unitful.ustrip(Unitful.NoUnits, Unitful.uconvert(Unitful.NoUnits, value / scale))
end

function _internalvalue(value::Base.RefValue, spec::PhysicalUnitSpec, scales::PhysicalScales, model, parameter::Symbol)
    return Ref(_internalvalue(value[], spec, scales, model, parameter))
end

function _internalvalue(value::AbstractArray, spec::PhysicalUnitSpec, scales::PhysicalScales, model, parameter::Symbol)
    return map(x -> _internalvalue(x, spec, scales, model, parameter), value)
end

function _internalvalue(value::Tuple, spec::PhysicalUnitSpec, scales::PhysicalScales, model, parameter::Symbol)
    return map(x -> _internalvalue(x, spec, scales, model, parameter), value)
end

function _internalvalue(value, spec::PhysicalUnitSpec, scales::PhysicalScales, model, parameter::Symbol)
    return value
end

"""
    physicalvalue(value, spec, scales, model)

Convert an internal numeric parameter value back to its physical scale.
"""
function physicalvalue(value, spec::PhysicalUnitSpec, scales, model; parameter::Symbol = :value)
    scale = _scale_product(physicalscales(scales), spec, model, parameter, value)
    return value .* scale
end

physicalvalue(value, ::Nothing, scales, model; parameter::Symbol = :value) = value

function _length_reference_scale(physical_scales, values...)
    if physical_scales isa PhysicalScales && !isnothing(physical_scales.length[])
        return physical_scales.length[]
    elseif physical_scales isa NamedTuple && haskey(physical_scales, :length)
        return physical_scales.length
    end

    found = _first_unitful(values...)
    isnothing(found) && return nothing
    inferred_scale = oneunit(found)
    if physical_scales isa PhysicalScales
        # A graph constructor passes a shared PhysicalScales object into the
        # topology constructor. Preserve the inferred topology unit there so
        # physical weight generators can reconstruct Unitful dr values later.
        physical_scales.length[] = inferred_scale
    end
    return inferred_scale
end

_first_unitful() = nothing
function _first_unitful(value, rest...)
    current = _first_unitful(value)
    return isnothing(current) ? _first_unitful(rest...) : current
end
_first_unitful(value::Unitful.AbstractQuantity) = value
_first_unitful(value::Tuple) = _first_unitful(value...)
_first_unitful(value::AbstractArray) = _first_unitful(Tuple(value)...)
_first_unitful(value) = nothing

function _internal_length_value(value::Unitful.AbstractQuantity, scale)
    isnothing(scale) && (scale = oneunit(value))
    return Unitful.ustrip(Unitful.NoUnits, Unitful.uconvert(Unitful.NoUnits, value / scale))
end

_internal_length_value(value, scale) = value

function _internal_length_container(value::Tuple, scale)
    return map(x -> _internal_length_container(x, scale), value)
end

function _internal_length_container(value::AbstractArray, scale)
    return map(x -> _internal_length_container(x, scale), value)
end

function _internal_length_container(value, scale)
    return _internal_length_value(value, scale)
end
