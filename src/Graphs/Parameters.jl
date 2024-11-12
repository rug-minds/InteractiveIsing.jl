export param, params

param(g::IsingGraph, param::Symbol) = getparam(g, param)
params(g::IsingGraph) = g.params

function changeactivation!(g, param, activate)
    g.params = changeactivation(g.params, param, activate)
    refresh(g)
    return g.params
end



@inline function getparam(g::IsingGraph, param::Symbol)
    getproperty(g.params, param)
end



activate!(g::IsingGraph, param) = changeactivation!(g, param, true)
deactivate!(g::IsingGraph, param) = changeactivation!(g, param, false)
function setglobal!(g::IsingGraph, param, val)
    pval = getparam(g, param)
    old_default = default(pval)
    g.params = Parameters(param = ParamVal(pval, val, false); get_nt(g.params)...)
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
