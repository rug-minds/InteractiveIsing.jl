Base.@propagate_inbounds function genSPAdj(layer, wg)
    row_idxs, col_idxs, weights = genLayerConnections(layer, wg)
    old_row_idxs, old_col_idxs, old_weights = removeConnections(layer)

    append!(row_idxs,    old_row_idxs)
    append!(col_idxs,    old_col_idxs)
    append!(weights ,    old_weights)

    return row_idxs, col_idxs, weights
end

@inline genSPAdj!(layer, wg) = set_sp_adj!(graph(layer), genSPAdj(layer, wg))
export genSPAdj!


Base.@propagate_inbounds function genSPAdj(layer1, layer2, wg)
    @assert layer1 != layer2
    row_idxs, col_idxs, weights = genLayerConnections(layer1, layer2, wg)
    old_row_idxs, old_col_idxs, old_weights = removeConnections(layer1, layer2)

    append!(row_idxs,    old_row_idxs)
    append!(col_idxs,    old_col_idxs)
    append!(weights ,    old_weights)

    return row_idxs, col_idxs, weights
end
@inline genSPAdj!(layer1, layer2, wg) = set_sp_adj!(graph(layer1), genSPAdj(layer1, layer2, wg))

@inline removeConnections!(layer) = set_sp_adj!(graph(layer), removeConnections(layer))
@inline removeConnections!(layer1, layer2) = set_sp_adj!(graph(layer1), removeConnections(layer1, layer2))
export genSPAdj!, genSPAdj, removeConnections!, removeConnections