include("Utils.jl")
include("IndexManipulation.jl")
include("ParamVal.jl")
include("SparseVal.jl")
export Parameters, getparam, setparam!, _setparam!

struct Parameters{_NT<:NamedTuple}
    _nt::_NT
end

include("Macros.jl")
include("Summing.jl")
include("ParameterRefs/ParameterRefs.jl")

Base.fieldnames(ps::Parameters) = fieldnames(ps._nt)
Base.fieldnames(ps::Type{<:Parameters}) = fieldnames(ps.parameters[1])
Base.fieldtypes(ps::Type{<:Parameters}) = fieldtypes(ps.parameters[1])

@inline @generated function gettype(p::Type{<:Parameters}, ::Val{field}) where {field}
    NT = p.parameters[1]
    idx = findfirst(x -> x == field, fieldnames(NT))
    if idx === nothing
        error("Field $field not found in type $NT, with fields $(fieldnames(NT))")
    end
    return :($(fieldtypes(NT)[idx]))
end

Parameters(; kwargs...) = Parameters(NamedTuple(kwargs))

@inline get_nt(@specialize(p::Parameters{NT})) where NT = getfield(p, :_nt)
@inline get_nt(::Type{IP}) where IP <: Parameters  = IP.parameters[1]
@inline get_nt(::Type{NT}) where NT <: NamedTuple = NT
export get_nt
#Forward methods for namedtuple to Parameters
# Base.getindex(p::Parameters, k::Symbol) = get_nt(p)[k]
Base.getindex(p::Parameters, val) = get_nt(p)[val]
Base.get!(p::Parameters, k::Symbol, default) = get(get_nt(p), k, default)
Base.get(p::Parameters, k::Symbol) = get(get_nt(p), k)
Base.get(p::Parameters, k::Symbol, default) = get(get_nt(p), k, default)
Base.length(p::Parameters) = length(get_nt(p)) 
Base.keys(p::Parameters) = keys(get_nt(p))
Base.values(p::Parameters) = values(get_nt(p))
Base.iterate(p::Parameters, state = 1) = iterate(get_nt(p), state)
Base.haskey(p::Parameters, k::Symbol) = haskey(get_nt(p), k)

function Base.getproperty(p::Parameters{NT}, s::Symbol) where {NT}
    @assert s != :_nt "Cannot access _nt directly, access through get_nt(params)"
    return getproperty(get_nt(p), s)
end

# isactive(p::Parameters, param::Symbol) = isactive(p::Parameters, Val{param}())
function isactive(p::Parameters, s::Symbol)
    return isactive(getproperty(p, s))::Bool
end

function isactive(::Type{IP}, s::Symbol) where IP
    NT = get_nt(IP)
    return isactive(gettype(NT, s))::Bool
end

function default(p::Parameters, s::Symbol)
    return default(getproperty(p, s))
end

function default(::Type{IP}, s::Symbol) where IP
    NT = get_nt(IP)
    return default(gettype(NT, s))
end

function description(p::Parameters, s::Symbol)
    return description(getproperty(p, s))
end

export default, isactive

# Implement pairs for correct splatting behavior
Base.pairs(p::Parameters) = pairs(get_nt(p))

# If you need mutability, implement setindex! and setproperty!
# Note: This will only work if the underlying NamedTuple is mutable
Base.setindex!(p::Parameters, v, k::Symbol) = setindex!(get_nt(p), v, k)
Base.setproperty!(p::Parameters, s::Symbol, v) = setproperty!(get_nt(p), s, v)

# To make it behave more like a NamedTuple in other contexts
Base.:(==)(a::Parameters, b::Parameters) = get_nt(a) == get_nt(b)
Base.hash(p::Parameters, h::UInt) = hash(get_nt(p), h)

# For pretty printing
Base.show(io::IO, p::Parameters) = print(io, "Parameters", get_nt(p))

# To allow splatting directly
Base.splat(p::Parameters) = splat(get_nt(p))

function changeactivation(params, param, activate)
    if !isnothing(activate) && isactive(params, param) != activate
        newparam = ParamVal(getproperty(params,param).val, default(params, param), activate, description = description(params, param))
        return Parameters(;get_nt(params)..., param => newparam)
    end 
    return params
end

function changedefault(params, param, val)
    if default(params, param) != val
        newparam = ParamVal(getproperty(params,param).val, val, isactive(params, param), description = description(params, param))
        return Parameters(;get_nt(params)..., param => newparam)
    end 
    return params
end

activate(params, param) = changeactivation(params, param, true)
deactivate(params, param) = changeactivation(params, param, false)

function push(params; param...)
    for key in keys(param)
        if haskey(params, key)
            @warn "Parameter $key already exists in the parameters, overwriting"
        end
    end
    return Parameters(;param..., get_nt(params)...)
end
include("Expressions.jl")

