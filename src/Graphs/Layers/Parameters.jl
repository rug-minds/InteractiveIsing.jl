# SET AND GET GRAPH PARAMETERS THROUGH THE LAYER

function getparam(l::AbstractIsingLayer, param::Symbol)
    val = getparam(graph(l), param)
    paramtype = eltype(val)
    if paramtype <: Vector
        return val[graphidxs(l)]
    else
        return val
    end
end

function setparam!(l::IsingLayer, param::Symbol, val, active = true) 
    return setparam!(graph(l), param, val, active, startidx(l), endidx(l))
end