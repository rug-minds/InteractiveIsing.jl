function spin_idx_to_layer_idx(j, layers)
    range_ends = range_end.(layers)
    return tuple_searchsortedfirst(range_ends, j)
end

"""
Dispatch a function statically and inlineable to a layer from an index
    This means we're basically doing an simple switch statement
    if idx in layer1
        func_to_dispatch(layer1)
    elseif idx in layer2
        func_to_dispatch(layer2)
    ...
    else        
        error("Index out of bounds for layers")
    end
"""
@inline Base.@constprop :aggressive function inline_layer_dispatch(func_to_dispatch::F, layer_idx::Int, layers::T) where {F,T}
    if layer_idx == 1
        return @inline func_to_dispatch(gethead(layers))
    else
        return @inline inline_layer_dispatch(func_to_dispatch, layer_idx - 1, gettail(layers))
    end
end

@inline function spin_idx_layer_dispatch(func_to_dispatch::F, j, layers::T) where {F,T}
    layer_idx = spin_idx_to_layer_idx(j, layers)
    return @inline inline_layer_dispatch(func_to_dispatch, layer_idx, layers)
end

@inline function spin_idx_layer_dispatch(func_to_dispatch::F, j, g::AbstractIsingGraph) where {F}
    return @inline spin_idx_layer_dispatch(func_to_dispatch, j, layers(g))
end