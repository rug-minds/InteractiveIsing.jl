@inline Base.@constprop :aggressive function inline_layer_dispatch(func_to_dispatch, layer_idx::Int, layers::T) where {T}
    if layer_idx == 1
        return @inline func_to_dispatch(gethead(layers))
    else
        return @inline inline_layer_dispatch(func_to_dispatch, layer_idx - 1, gettail(layers))
    end
end