export getParam, setParam!, _setparam!
@inline function getParam(g::IsingGraph, param::Symbol)
    @assert haskey(g.params, param) "Parameter $param not found in graph"
    return g.params[param]
end

function changeActivation!(g, param, activate)
    if !isnothing(activate) && isactive(g.params[param]) != activate
        newparam = ParamVal(g.params[param], activate)
        println("Newparam, ", newparam)
        println("Newparams: ", (;g.params..., param => newparam))
        g.params = (;g.params..., param => newparam)
        refresh(g)
    end 
end

activate!(g::IsingGraph, param) = changeActivation!(g, param, true)
deactivate!(g::IsingGraph, param) = changeActivation!(g, param, false)
function setGlobal!(g::IsingGraph, param, val)
    pval = getparam(g, param)
    old_default = default(pval)
    g.params = (;g.params..., param => ParamVal(pval, val, false))
    if old_default != val
        refresh(g)
    end
    return g.params[param]
end

"""
Get a graph and a symbol, then set the value of the parameter to the given value
"""
function setParam!(g::IsingGraph, param::Symbol, val, active = nothing, si = nothing, ei = nothing)
    @assert haskey(g.params, param) "Parameter $param not found in graph"
    pval = g.params[param]
    _setparam!(pval, param, val, si, ei)

    changeActivation!(g, param, active)
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