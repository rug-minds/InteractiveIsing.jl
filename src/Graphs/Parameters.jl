export getparam, getparams

getparams(g::AbstractIsingGraph) = params(g)
getparam(g::AbstractIsingGraph, param::Symbol) = h_param(g, param)

param(g::AbstractIsingGraph, param::Symbol) = getparam(g, param)
params(g::AbstractIsingGraph) = g.params

function changeactivation!(g, param, activate)
    old_active = isactive(getparam(g, param))
    g.params = changeactivation(g.params, param, activate)
    if old_active != activate
        reprepare(g)
    end
    return g.params
end

activate!(g::AbstractIsingGraph, param) = changeactivation!(g, param, true)
deactivate!(g::AbstractIsingGraph, param) = changeactivation!(g, param, false)
function setglobal!(g::AbstractIsingGraph, param, val)
    pval = getparam(g, param)
    old_default = default(pval)
    g.params = Parameters(param = ParamTensor(pval, val, false); get_nt(g.params)...)
    if old_default != val
        reprepare(g)
    end
    return g.params[param]
end

# TODO: Standard behavior should be to turn it on?
"""
Get a graph and a symbol, then set the value of the parameter to the given value
"""
function setparam!(g::AbstractIsingGraph, param::Symbol, val, active = true, si = nothing, ei = nothing)
    @assert haskey(g.params, param) "Parameter $param not found in graph"
    pval = g.params[param]
    _setparam!(pval, param, val, si, ei)

    changeactivation!(g, param, active)
    return g.params[param]
end

function _setparam!(pval::ParamTensor{T}, param::Symbol, val, si = nothing, ei = nothing) where T
    param[] = val
    return nothing
end

function _setparam!(pval::ParamTensor{T}, param::Symbol, val, startidx = nothing, endidx = nothing) where T<:Vector
    isnothing(startidx) && (startidx = 1)
    isnothing(endidx)   && (endidx = length(pval))
    @assert length(pval) >= endidx - startidx + 1
    pval.val[startidx:endidx] .= val
    return nothing
end
