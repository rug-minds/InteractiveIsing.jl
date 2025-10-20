export genAdj!, genAdj, remAdj!, genAdjFull!, remAdjAll!, viewConnections

"""
Generate the connections in a layer based on a weightgenerator
Returns sparse representation of the connections
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
Then put the connections directly in the graph
"""
@inline genAdj!(layer::AbstractIsingLayer, wg) = set_adj!(layer, wg, genAdj(layer, wg))
@inline genAdj!(x::AbstractIsingLayer, ::Nothing) = nothing

"""
Give a weightgenerator and two layers.
Return the connections between the two layers in sparse format
i.e. the rows, columns and weights of the connections
"""
Base.@propagate_inbounds function genAdj(layer1, layer2, wg)
    @assert layer1 != layer2
    row_idxs, col_idxs, weights = genLayerConnections(layer1, layer2, wg)
    old_row_idxs, old_col_idxs, old_weights = removeConnections(layer1, layer2)

    append!(row_idxs,    old_row_idxs)
    append!(col_idxs,    old_col_idxs)
    append!(weights ,    old_weights)

    return row_idxs, col_idxs, weights
end
"""
Fully connect two layers
"""
genAdjFull!(l1, l2) = connectLayersFull!(l1,l2)

"""
Generate the connections between two layers based on a weightgenerator and set the connections in the layers
Then put the connections directly in the graph
"""
@inline genAdj!(layer1::AbstractIsingLayer, layer2::AbstractIsingLayer, wg) = set_adj!(layer1, layer2, wg, genAdj(layer1, layer2, wg))

"""
Removes the connections within a layer
"""
@inline remAdj!(layer) = set_adj!(graph(layer), removeConnections(layer))
"""
Removes the connections between two specified layers
"""
@inline remAdj!(layer1, layer2) = set_adj!(graph(layer1), removeConnections(layer1, layer2))

"""
Removes all connections in a layer
"""
remAdjAll!(layer) = set_adj!(graph(layer), removeConnectionsAll(layer))


"""
Give two layers and an index in the first layer
Return the connections of the index in the first layer to the second layer
    by returning the coordinates of the connections and the values of the connections
"""
function viewConnections(layer1::AbstractIsingLayer, layer2::AbstractIsingLayer, idx)
    # g = graph(layer1)
    # g_idx = idxLToG(idx, layer)
    connections = adj(layer1)[:, idx]
    idxs = findall(x -> x ∈ graphidxs(layer2), connections.nzind)
    return idxToCoord.(idxGToL.(connections.nzind[idxs], Ref(layer2)), Ref(layer2)), connections.nzval[idxs]
end

"""
Same as the other, but now provide the coordinates of the index in the first layer
"""
viewConnections(layer1::AbstractIsingLayer, layer2::AbstractIsingLayer, i, j) = viewConnections(layer1, layer2, coordToIdx(i,j,layer1))

"""
Give a list of indexes in the state of a layer
Return the rows and columns of the indexes
"""
function idxs2rowscols(layer, idxs)
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

