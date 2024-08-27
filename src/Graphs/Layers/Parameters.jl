# SET AND GET GRAPH PARAMETERS THROUGH THE LAYER

function getParam(l, param::Symbol)
    val = getParam(graph(l), param)
    paramtype = eltype(val)
    if paramtype <: Vector
        return val[graphidxs(l)]
    else
        return val
    end
end

function setParam!(l::IsingLayer, param::Symbol, val, active = nothing) 
    return setParam!(graph(l), param, val, active, startidx(l), endidx(l))
end