export genAdj!, removeConnections!, removeConnectionsAll!, viewConnections

"""
Generate the connections in a layer based on a weightgenerator
"""
Base.@propagate_inbounds function genAdj(layer, wg)
    row_idxs, col_idxs, weights = genLayerConnections(layer, wg)
    old_row_idxs, old_col_idxs, old_weights = removeConnections(layer)

    append!(row_idxs,    old_row_idxs)
    append!(col_idxs,    old_col_idxs)
    append!(weights ,    old_weights)

    return row_idxs, col_idxs, weights
end
"""
Generate the connections in a layer based on a weightgenerator and set the connections in the layer
"""
@inline genAdj!(layer::IsingLayer, wg) = set_adj!(layer, wg, genAdj(layer, wg))


Base.@propagate_inbounds function genAdj(layer1, layer2, wg)
    @assert layer1 != layer2
    row_idxs, col_idxs, weights = genLayerConnections(layer1, layer2, wg)
    old_row_idxs, old_col_idxs, old_weights = removeConnections(layer1, layer2)

    append!(row_idxs,    old_row_idxs)
    append!(col_idxs,    old_col_idxs)
    append!(weights ,    old_weights)

    return row_idxs, col_idxs, weights
end
@inline genAdj!(layer1::IsingLayer, layer2::IsingLayer, wg) = set_adj!(layer1, layer2, wg, genAdj(layer1, layer2, wg))

@inline removeConnections!(layer) = set_adj!(graph(layer), removeConnections(layer))
@inline removeConnections!(layer1, layer2) = set_adj!(graph(layer1), removeConnections(layer1, layer2))

removeConnectionsAll!(layer) = set_adj!(graph(layer), removeConnectionsAll(layer))

export genAdj!, removeConnections!, removeConnectionsAll!

# viewConnections(layer::IsingLayer, i,j) = viewConnections(layer, coordToIdx(i,j,layer))

function viewConnections(layer1::IsingLayer, layer2::IsingLayer, idx)
    # g = graph(layer1)
    # g_idx = idxLToG(idx, layer)
    connections = adj(layer1)[:, idx]
    idxs = findall(x -> x ∈ graphidxs(layer2), connections.nzind)
    return idxToCoord.(idxGToL.(connections.nzind[idxs], Ref(layer2)), Ref(layer2)), connections.nzval[idxs]
end

viewConnections(layer1::IsingLayer, layer2::IsingLayer, i, j) = viewConnections(layer1, layer2, coordToIdx(i,j,layer1))

function idxs2rowscols(idxs)
    rows = []
    cols = []
    sizehint!(rows, length(idxs))
    sizehint!(cols, length(idxs))

    for idx in idxs
        row, col = idxToCoord(idx, glength(layer))
        push!(rows, row)
        push!(cols, col)
    end
    return rows, cols
end

export viewConnections