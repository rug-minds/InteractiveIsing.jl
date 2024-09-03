export getparam, setparam!, _setparam!

struct IsingParameters{NT<:NamedTuple}
    nt::NT
end
IsingParameters(; kwargs...) = IsingParameters(NamedTuple(kwargs))

#Forward methods for namedtuple to IsingParameters
Base.getindex(p::IsingParameters, k::Symbol) = p.nt[k]
Base.get!(p::IsingParameters, k::Symbol, default) = get(p.nt, k, default)
Base.get(p::IsingParameters, k::Symbol) = get(p.nt, k)
Base.get(p::IsingParameters, k::Symbol, default) = get(p.nt, k, default)
Base.length(p::IsingParameters) = length(p.nt)
Base.keys(p::IsingParameters) = keys(p.nt)
Base.values(p::IsingParameters) = values(p.nt)
Base.iterate(p::IsingParameters, state = 1) = iterate(p.nt, state)
Base.haskey(p::IsingParameters, k::Symbol) = haskey(p.nt, k)

# Implement pairs for correct splatting behavior
Base.pairs(p::IsingParameters) = pairs(p.nt)

# If you need mutability, implement setindex! and setproperty!
# Note: This will only work if the underlying NamedTuple is mutable
Base.setindex!(p::IsingParameters, v, k::Symbol) = setindex!(p.nt, v, k)
Base.setproperty!(p::IsingParameters, s::Symbol, v) = setproperty!(p.nt, s, v)

# To make it behave more like a NamedTuple in other contexts
Base.:(==)(a::IsingParameters, b::IsingParameters) = a.nt == b.nt
Base.hash(p::IsingParameters, h::UInt) = hash(p.nt, h)

# For pretty printing
Base.show(io::IO, p::IsingParameters) = print(io, "IsingParameters", p.nt)

# To allow splatting directly
Base.splat(p::IsingParameters) = splat(p.nt)





@inline function getparam(g::IsingGraph, param::Symbol)
    @assert haskey(g.params, param) "Parameter $param not found in graph"
    return g.params[param]
end

function changeactivation!(g, param, activate)
    if !isnothing(activate) && isactive(g.params[param]) != activate
        newparam = ParamVal(g.params[param], activate)
        # println("Newparam, ", newparam)
        # println("Newparams: ", (;g.params..., param => newparam))
        g.params = IsingParameters(param = newparam; g.params.nt...)
        refresh(g)
    end 
end

activate!(g::IsingGraph, param) = changeactivation!(g, param, true)
deactivate!(g::IsingGraph, param) = changeactivation!(g, param, false)
function setglobal!(g::IsingGraph, param, val)
    pval = getparam(g, param)
    old_default = default(pval)
    g.params = IsingParameters(param = ParamVal(pval, val, false); g.params.nt...)
    if old_default != val
        refresh(g)
    end
    return g.params[param]
end

# TODO: Standard behavior should be to turn it on?
"""
Get a graph and a symbol, then set the value of the parameter to the given value
"""
function setparam!(g::IsingGraph, param::Symbol, val, active = nothing, si = nothing, ei = nothing)
    @assert haskey(g.params, param) "Parameter $param not found in graph"
    pval = g.params[param]
    _setparam!(pval, param, val, si, ei)

    changeactivation!(g, param, active)
    return g.params[param]
end

function _setparam!(pval::ParamVal{T}, param::Symbol, val, si = nothing, ei = nothing) where T
    param[] = val
    return nothing
end

function _setparam!(pval::ParamVal{T}, param::Symbol, val, startidx = nothing, endidx = nothing) where T<:Vector
    isnothing(startidx) && (startidx = 1)
    isnothing(endidx)   && (endidx = length(pval[]))
    @assert length(pval[]) >= endidx - startidx + 1
    pval.val[startidx:endidx] .= val
    return nothing
end