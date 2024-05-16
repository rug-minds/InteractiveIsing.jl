function getparam(l, param::Symbol)
    params = graph(l).params
    @assert haskey(params, param) "Parameter $param not found in graph"
    paramtype = eltype(params[param])
    if paramtype <: Vector
        return params[param][graphidxs(l)]
    else
        return params[param]
    end
end

function set_param!(l::IsingLayer, param::Symbol, val, active = nothing) 
   params = graph(l).params
   @assert haskey(params, param) "Parameter $param not found in graph"
   _set_param!(l, params, param, val)
    if !isnothing(active)
        g.params = (;g.params..., param => ParamVal(val, active))
    end
end

function _set_param!(l, params, param::Symbol, val)
    params[param] = val
    return nothing
end

function _set_param!(l, params, param, val::Vector)
    @assert length(val) == length(state(l))
    params[param][graphidxs(l)] .= val 
    return nothing
end

function set_param!(g, param, val, idx, active = nothing)
    @assert haskey(g.params, param) "Parameter $param not found in graph"
    _set_param!(g.params, param, val, idx)
    if !isnothing(active)
        g.params = (;g.params..., param => ParamVal(val, active))
        notify(emitter(g))
    end
end

function _set_param!(params, param::Symbol, val, idx)
    params[param] = val
    return nothing
end

function _set_param!(params, param, val::Vector, idx)
    @assert length(val) == length(state(l))
    params[param][idx] .= val 
    return nothing
end