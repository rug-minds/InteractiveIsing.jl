"""
Optional template layer for Hamiltonian terms.

The real Hamiltonian interface is still `calculate`. This file only provides
an opt-in constructor convention for terms that want graph-instantiated
parameters.

Conventions:

- `ParameterSpec` is pre-instantiation.
- `Parameter` is post-instantiation.
- `Parameters` stores either specs or instantiated parameters in a named tuple.
- `Parameters.info` stores metadata separately from the values.
- `Parameters.units` stores optional unit metadata separately from the values.
- parameter origin is an instantiation truth, stored on `Parameter`, not on
  `ParameterSpec`.
"""

export ParameterSpec,
       Parameter,
       Parameters,
       ParameterOrigin,
       Passed,
       Defaulted,
       Derived,
       Linked,
       NoEnsure,
       Force,
       InternalImplementation,
       InternalPlan,
       ArrayPlan,
       TypePlan,
       ensure_isinggraph_eltype,
       ensure_isinggraph_scalar,
       ensure_isinggraph_state_length,
       ensure_isinggraph_state_vector,
       ensure_isinggraph_adjacency,
       parameter,
       value,
       origin,
       parameters,
       internal,
       instantiate

abstract type ParameterOrigin end
struct Passed <: ParameterOrigin end
struct Defaulted <: ParameterOrigin end
struct Derived <: ParameterOrigin end
struct Linked <: ParameterOrigin end

struct NoEnsure{T}
    value::T
end

struct Force{T}
    value::T
end

abstract type InternalImplementation end

struct ArrayPlan{F} <: AbstractArray{Any,0}
    f::F
end

struct TypePlan{F}
    f::F
end

struct InternalPlan{F,Values}
    f::F
    values::Values
end

Base.size(::ArrayPlan) = ()
Base.getindex(::ArrayPlan) =
    throw(ArgumentError("ArrayPlan is a pre-instantiation internal placeholder. Call instantiate(plan, model) first."))

InternalPlan(values::NamedTuple, f) = InternalPlan(f, values)

struct ParameterSpec{Default,DefaultType,Ensure,Check,Warn,Input,Units}
    # Front-end keyword name, e.g. :b for MagField(; b = ...).
    name::Symbol
    # Required type after instantiation unless the value is wrapped in Force.
    type::Type
    # Constructor input before graph instantiation. `nothing` means use `default`.
    input::Input
    # Value/template used when `input === nothing`.
    default::Default
    # Storage type used when explicit scalar/singleton input must be filled.
    # `nothing` means infer it from `default`.
    default_type::DefaultType
    # Normalization pipeline. Called as ensure(input_or_default, default, model).
    ensure::Ensure
    # Hard validation after ensure. Return false or throw to reject.
    check::Check
    # Soft validation after ensure/check. Return false to emit a warning.
    warn::Warn
    # Author-facing metadata kept separately in Parameters.info.
    info::String
    # Optional unit metadata kept separately in Parameters.units.
    units::Units
end

struct ConversionDefault{Storage,Default}
    default::Default
end

struct Parameter{Origin<:ParameterOrigin,Value}
    # Instantiated value used by calculate/update code.
    value::Value
end

Parameter(value::Value, origin::Origin) where {Value,Origin<:ParameterOrigin} =
    Parameter{Origin,Value}(value)

value(parameter::Parameter) = getfield(parameter, :value)
origin(::Parameter{Origin}) where {Origin<:ParameterOrigin} = Origin()

_noensure(x, default, model) = x
_nocheck(x, model) = nothing
_nowarn(x, model) = nothing

function _default_value(x)
    if x isa Number
        return x
    elseif x isa Base.RefValue
        return x[]
    elseif x isa AbstractArray && Base.ndims(x) == 0
        return x[]
    else
        return x
    end
end
_default_value(x::ConversionDefault) = _default_value(x.default)

function _storage_type(x)
    x isa Type && return x
    x isa Function &&
        throw(ArgumentError("Cannot infer Hamiltonian parameter fill storage from graph-derived default $(typeof(x)). Set `default_type` in the parameter spec."))
    return Base.typename(typeof(x)).wrapper
end
_storage_type(::ConversionDefault{Storage}) where {Storage} = Storage

function _fill_storage(::Type{Storage}, val, dims::Integer...) where {Storage}
    applicable(filltype, Storage, val, dims...) && return filltype(Storage, val, dims...)
    throw(ArgumentError("Hamiltonian parameter auto-fill requires `filltype(::Type{$(Storage)}, value, dims...)`. Define that method for custom storage type $(Storage)."))
end

function ensure_isinggraph_eltype(input, default, model)
    T = eltype(model)

    if input isa Number
        return convert(T, input)
    elseif input isa Base.RefValue
        return Ref(convert(T, input[]))
    elseif input isa AbstractArray
        return map(T, input)
    else
        return input
    end
end

function ensure_isinggraph_scalar(input, default, model)
    input isa DerivedParameter && return ensure_isinggraph_scalar(input(model), default, model)
    input isa Type && return _fill_storage(input, convert(eltype(model), _default_value(default)))
    applicable(input, model) && return ensure_isinggraph_scalar(input(model), default, model)

    T = eltype(model)

    if input isa Number
        return _fill_storage(_storage_type(default), convert(T, input))
    elseif input isa Base.RefValue || (input isa AbstractArray && Base.ndims(input) == 0)
        return _fill_storage(_storage_type(input), convert(T, _default_value(input)))
    else
        return input
    end
end

function ensure_isinggraph_state_length(input, default, model)
    # LEGACY / DEPRECATED:
    # `StateLike(...)` and other `DerivedParameter`s are old-style graph-resolved
    # Hamiltonian parameter inputs. New templates should express this through a
    # normal default plus ensure functions:
    #
    #     default = ConstFill(0)
    #     ensure = (ensure_isinggraph_state_length, ensure_isinggraph_eltype)
    #
    # This branch is kept temporarily for front-end/back-end compatibility and
    # should be removed after old docs/tutorials have migrated.
    if input isa DerivedParameter
        return ensure_isinggraph_state_length(input(model), default, model)
    end

    len = statelen(model)
    input isa Type && return _fill_storage(input, _default_value(default), len)
    applicable(input, model) && return ensure_isinggraph_state_length(input(model), default, model)

    if input isa Number
        return _fill_storage(_storage_type(default), input, len)
    elseif input isa Base.RefValue
        return _fill_storage(_storage_type(default), _default_value(input), len)
    elseif input isa AbstractArray && Base.ndims(input) == 0
        return _fill_storage(_storage_type(input), _default_value(input), len)
    elseif input isa AbstractArray
        length(input) == 1 && return _fill_storage(_storage_type(default), only(input), len)
        length(input) == len ||
            throw(DimensionMismatch("Expected state-like parameter with length $(len); got $(length(input))."))
        return input
    else
        return input
    end
end

function _deprecated_derived_parameter_message(name::Symbol, input)
    return "Hamiltonian parameter `$(name)` uses deprecated graph-derived input $(typeof(input)). This usually comes from a constructor keyword like `$(name) = StateLike(...)`. Pass a value, container, or graph function directly instead."
end

function _deprecated_derived_parameter_message(name::Symbol, input::StateLike{T}) where {T}
    value = repr(input.default_el)
    if T === ConstFill
        return "Hamiltonian parameter `$(name)` was passed `StateLike(ConstFill, $(value))`, probably from `$(name) = StateLike(ConstFill, $(value))` in the Hamiltonian constructor. Replace it with `$(name) = $(value)` for the term's default constant storage, or `$(name) = ConstFill($(value))` if you want to request `ConstFill` explicitly."
    elseif T === UniformArray
        return "Hamiltonian parameter `$(name)` was passed `StateLike(UniformArray, $(value))`, probably from `$(name) = StateLike(UniformArray, $(value))` in the Hamiltonian constructor. Replace it with `$(name) = UniformArray($(value))` for mutable uniform storage."
    else
        return "Hamiltonian parameter `$(name)` was passed `StateLike($(nameof(T)), $(value))`, probably from `$(name) = StateLike($(nameof(T)), $(value))` in the Hamiltonian constructor. Replace it with a direct graph function such as `$(name) = g -> filltype($(nameof(T)), $(value), statelen(g))`, or pass a concrete container."
    end
end

function ensure_isinggraph_state_vector(input, default, model)
    input isa DerivedParameter && return ensure_isinggraph_state_vector(input(model), default, model)

    T = eltype(model)
    len = statelen(model)
    input isa Type && return _fill_storage(input, convert(T, _default_value(default)), len)
    applicable(input, model) && return ensure_isinggraph_state_vector(input(model), default, model)

    if input isa Number
        return _fill_storage(_storage_type(default), convert(T, input), len)
    elseif input isa Base.RefValue || (input isa AbstractArray && Base.ndims(input) == 0)
        return _fill_storage(_storage_type(default), convert(T, _default_value(input)), len)
    elseif input isa AbstractArray
        length(input) == 1 && return _fill_storage(_storage_type(default), convert(T, only(input)), len)
        length(input) == len ||
            throw(DimensionMismatch("Expected state-like vector with length $(len); got $(length(input))."))
        return collect(T, input)
    else
        return input
    end
end

function ensure_isinggraph_adjacency(input, default, model)
    !(input isa Type) && applicable(input, model) && return ensure_isinggraph_adjacency(input(model), default, model)

    n = statelen(model)
    if input isa AbstractMatrix
        size(input, 1) == n && size(input, 2) == n ||
            throw(DimensionMismatch("Adjacency matrix size must match graph state length; expected $(n)x$(n), got $(size(input))."))
    end

    return input
end

function _apply_ensure(ensure, input, default, model)
    return ensure(input, default, model)
end

function _apply_ensure(ensures::Tuple, input, default, model)
    value = input
    for ensure in ensures
        value = _apply_ensure(ensure, value, default, model)
    end
    return value
end

"""
    parameter(; x = nothing, type = Any, default = nothing, default_type = nothing, ensure, check, warn, info = "", units = nothing)

Create a pre-instantiation parameter spec from constructor syntax. Exactly one
non-template keyword is expected; its name becomes the parameter name.

`ensure` is called as `ensure(input, default, model)`.
"""
function parameter(; type = Any,
                     default = nothing,
                     default_type = nothing,
                     ensure = _noensure,
                     check = _nocheck,
                     warn = _nowarn,
                     info = "",
                     units = nothing,
                     kwargs...)
    length(kwargs) == 1 ||
        throw(ArgumentError("Expected exactly one parameter keyword, got $(length(kwargs))."))

    name = first(keys(kwargs))
    input = first(values(kwargs))

    return ParameterSpec(name, type, input, default, default_type, ensure, check, warn, info, units)
end

"""
Hamiltonian parameter storage.

Before instantiation, `entries` should contain `ParameterSpec`s.
After instantiation, `entries` should contain `Parameter`s.
`info` and `units` are named tuples keyed by parameter name and kept separate
from values.
"""
struct Parameters{Entries,Info,Units}
    # NamedTuple of ParameterSpec before instantiation, Parameter after.
    entries::Entries
    # NamedTuple of metadata keyed by parameter name.
    info::Info
    # NamedTuple of unit metadata keyed by parameter name.
    units::Units
end

function Parameters(entries::ParameterSpec...; info = (;), units = (;))
    named_entries = (;)
    named_info = info
    named_units = units

    for entry in entries
        haskey(named_entries, entry.name) &&
            throw(ArgumentError("Duplicate Hamiltonian parameter name :$(entry.name)."))
        named_entries = (; named_entries..., entry.name => entry)
        haskey(named_info, entry.name) ||
            (named_info = (; named_info..., entry.name => entry.info))
        haskey(named_units, entry.name) ||
            (named_units = (; named_units..., entry.name => entry.units))
    end

    return Parameters(named_entries, named_info, named_units)
end

Parameters(entries::NamedTuple; info = (;), units = (;)) = Parameters(entries, info, units)

Base.keys(params::Parameters) = keys(getfield(params, :entries))
Base.length(params::Parameters) = length(keys(params))
Base.pairs(params::Parameters) = pairs(getfield(params, :entries))
Base.iterate(params::Parameters, state...) = iterate(values(getfield(params, :entries)), state...)

function Base.propertynames(params::Parameters; private = false)
    names = keys(getfield(params, :entries))
    return private ? (:entries, :info, :units, names...) : names
end

function Base.getproperty(params::Parameters, name::Symbol)
    name === :entries && return getfield(params, :entries)
    name === :info && return getfield(params, :info)
    name === :units && return getfield(params, :units)

    entries = getfield(params, :entries)
    if haskey(entries, name)
        entry = getproperty(entries, name)
        return entry isa Parameter ? value(entry) : entry
    end

    return getfield(params, name)
end

_origin(input) = isnothing(input) ? Defaulted() : Passed()
_resolve_default(default, model) = applicable(default, model) ? default(model) : default

"""
    _physical_parameter_value(spec, value, model)

Convert Unitful constructor input for one Hamiltonian parameter before the
ordinary `ensure` pipeline normalizes storage and graph precision.
"""
function _physical_parameter_value(spec::ParameterSpec, value, model)
    return internalvalue(value, spec.units, physicalscales(model), model; parameter = spec.name)
end

function _conversion_default(spec::ParameterSpec)
    isnothing(spec.default_type) && return spec.default
    return ConversionDefault{spec.default_type, typeof(spec.default)}(spec.default)
end

function _run_check(spec::ParameterSpec, value, model)
    result = spec.check(value, model)
    result === false && throw(ArgumentError("Check failed for Hamiltonian parameter :$(spec.name)."))
    return nothing
end

function _run_warn(spec::ParameterSpec, value, model)
    result = spec.warn(value, model)
    result === false && @warn "Hamiltonian parameter warning." parameter = spec.name
    return nothing
end

instantiate(x, model) = x
instantiate(::Nothing, model) = nothing
instantiate(plan::ArrayPlan, model) = plan.f(model)
instantiate(plan::TypePlan, model) = plan.f(model)
instantiate(plan::InternalPlan, model) = plan.f(plan, model)

function instantiate(internals::InternalImplementation, model)
    constructor = Base.typename(typeof(internals)).wrapper
    values = map(fieldnames(typeof(internals))) do name
        instantiate(getfield(internals, name), model)
    end
    return constructor(values...)
end

function instantiate(spec::ParameterSpec, model)
    input = spec.input
    param_origin = _origin(input)
    input isa DerivedParameter && @warn _deprecated_derived_parameter_message(spec.name, input) maxlog = 1

    if isnothing(input)
        default_value = _physical_parameter_value(spec, _resolve_default(spec.default, model), model)
        resolved = _apply_ensure(spec.ensure, default_value, spec.default, model)
        run_check = true
    elseif input isa NoEnsure
        resolved = _physical_parameter_value(spec, input.value, model)
        run_check = true
    elseif input isa Force
        resolved = _physical_parameter_value(spec, input.value, model)
        run_check = false
    else
        converted = _physical_parameter_value(spec, input, model)
        resolved = _apply_ensure(spec.ensure, converted, _conversion_default(spec), model)
        run_check = true
    end

    if run_check
        resolved isa spec.type ||
            throw(ArgumentError("Invalid resolved value for Hamiltonian parameter :$(spec.name). Expected $(spec.type), got $(typeof(resolved))."))
        _run_check(spec, resolved, model)
    end

    _run_warn(spec, resolved, model)

    return Parameter(resolved, param_origin)
end

function instantiate(params::Parameters, model)
    instantiated = (;)

    for name in keys(params.entries)
        instantiated = (; instantiated..., name => instantiate(getproperty(params.entries, name), model))
    end

    return Parameters(instantiated; info = params.info, units = params.units)
end

_hasfield(x, name::Symbol) = name in fieldnames(typeof(x))
_hasfield(::Type{T}, name::Symbol) where {T} = name in fieldnames(T)

parameters(T::Type) = T.parameters

parameters(ham::HamiltonianTerm) =
    _hasfield(ham, :parameters) ? getfield(ham, :parameters) : nothing

internal(ham::HamiltonianTerm) =
    _hasfield(ham, :internal) ? getfield(ham, :internal) : nothing

function _template_entry_names(::Type{H}) where {H}
    hasfield(H, :parameters) || return ()

    P = fieldtype(H, :parameters)
    P <: Parameters || return ()

    entries_type = P.parameters[1]
    entries_type <: NamedTuple || return ()
    return fieldnames(entries_type)
end

@generated function _hamiltonianterm_propertynames(::Type{H}) where {H<:HamiltonianTerm}
    fields = fieldnames(H)
    nested = Tuple(name for name in _template_entry_names(H) if !(name in fields))
    return :($(QuoteNode((fields..., nested...))))
end

Base.propertynames(ham::HamiltonianTerm; private = false) =
    _hamiltonianterm_propertynames(typeof(ham))

function Base.getproperty(ham::HamiltonianTerm, name::Symbol)
    _hasfield(ham, name) && return getfield(ham, name)

    if _hasfield(ham, :parameters)
        params = getfield(ham, :parameters)
        if params isa Parameters && haskey(getfield(params, :entries), name)
            return getproperty(params, name)
        end
    end

    if _hasfield(ham, :internal)
        internals = getfield(ham, :internal)
        if !isnothing(internals) && hasfield(typeof(internals), name)
            return getfield(internals, name)
        end
    end

    return getfield(ham, name)
end

function HamiltonianTerm(::Type{H}, params, internals) where {H<:HamiltonianTerm}
    hasparams = _hasfield(H, :parameters)
    hasinternal = _hasfield(H, :internal)
    constructor = Base.typename(H).wrapper

    if hasparams && hasinternal
        return constructor(params, internals)
    elseif hasparams
        return constructor(params)
    elseif hasinternal
        return constructor(internals)
    else
        throw(ArgumentError("Hamiltonian type $(H) does not use the term-template field convention."))
    end
end

function instantiate(ham::H, model) where {H<:HamiltonianTerm}
    (_hasfield(H, :parameters) || _hasfield(H, :internal)) || return ham
    return HamiltonianTerm(H, instantiate(parameters(ham), model), instantiate(internal(ham), model))
end

_template_showctx(io::IO) =
    isdefined(@__MODULE__, :_showctx) ? _showctx(io) : IOContext(io, :limit => get(io, :limit, false), :compact => get(io, :compact, false))

_template_truncate_for_prefix(io::IO, prefix::AbstractString, text::AbstractString) =
    isdefined(@__MODULE__, :_truncate_for_prefix) ? _truncate_for_prefix(io, prefix, text) : text

function _parameter_summary(entry)
    if entry isa Parameter
        return string(summary(value(entry)), " [", nameof(typeof(origin(entry))), "]")
    elseif entry isa ParameterSpec
        input = isnothing(entry.input) ? "default" : summary(entry.input)
        return string("spec(", input, " -> ", entry.type, ")")
    else
        return summary(entry)
    end
end

function _show_parameters(io::IO, params::Parameters; prefix = "")
    entries = getfield(params, :entries)
    isempty(entries) && (print(io, prefix, "(empty)"); return nothing)

    show_ctx = _template_showctx(io)
    last_idx = length(entries)
    for (idx, name) in enumerate(keys(entries))
        branch = idx == last_idx ? "└── " : "├── "
        stem = idx == last_idx ? "    " : "│   "
        line_prefix = string(prefix, branch)
        entry = getproperty(entries, name)
        value_text = _parameter_summary(entry)
        info = get(getfield(params, :info), name, "")
        units = get(getfield(params, :units), name, nothing)
        units_text = isnothing(units) ? "" : string(" {", units, "}")
        line = isempty(info) ? string(name, " = ", value_text, units_text) : string(name, " = ", value_text, units_text, " - ", info)
        println(io, line_prefix, _template_truncate_for_prefix(io, line_prefix, line))

        if entry isa Parameter
            value_prefix = string(prefix, stem, "└── ")
            value_ctx = IOContext(show_ctx, :limit => true)
            shown_value = sprint(show, value(entry); context = value_ctx, sizehint = 0)
            print(io, value_prefix, _template_truncate_for_prefix(io, value_prefix, string("value = ", shown_value)))
            idx == last_idx || println(io)
        end
    end
    return nothing
end

Base.show(io::IO, ::MIME"text/plain", params::Parameters) = _show_parameters(io, params)

function _show_templated_hamiltonian(io::IO, hterm::HamiltonianTerm)
    params = parameters(hterm)
    params isa Parameters || (show(io, hterm); return nothing)

    println(io, summary(hterm))
    println(io, "└── parameters")
    _show_parameters(io, params; prefix = "    ")
    return nothing
end

function Base.show(io::IO, mime::MIME"text/plain", hterm::HamiltonianTerm)
    if _hasfield(hterm, :parameters) && getfield(hterm, :parameters) isa Parameters
        return _show_templated_hamiltonian(io, hterm)
    end
    return invoke(show, Tuple{IO, MIME"text/plain", Any}, io, mime, hterm)
end
