initSPAdj!(g) = adj(g, spzeros(nStates(g), nStates(g)))
export initSPAdj!

"""
Struct to hold the connections in sparce format
    Not used?
"""
struct SparseConnections
    row_idxs::Vector{Int32}
    col_idxs::Vector{Int32}
    weights::Vector{Float32}
end


"""
Give layer and WeightGenerator
    returns the connections within the layer in row_idxs, col_idxs, and weights
"""
function genLayerConnections(layer, wg)
    row_idxs = Int32[]
    col_idxs = Int32[]
    weights = Float32[]

    g = graph(layer)

    _NN = wg.NN
    blocksize = (2*_NN+1)^2
    n_conns = nStates(g)*blocksize
    sizehint!(col_idxs, 2*n_conns)
    sizehint!(row_idxs, 2*n_conns)
    sizehint!(weights, 2*n_conns)

    pre_3tuple = Prealloc(NTuple{3, Int32}, blocksize)
    _fillSparseVecs(layer, row_idxs, col_idxs, weights, _NN, top(layer), wg, pre_3tuple, SelfType(wg))

    append!(row_idxs, col_idxs)
    append!(col_idxs, @view(row_idxs[1:end÷2]))
    append!(weights, weights)

    return row_idxs, col_idxs, weights
end

"""
Give layer and WeightGenerator
    returns the connections within the layer in row_idxs, col_idxs, and weights
"""
function genLayerConnections(layer1, layer2, wg)
    row_idxs = Int32[]
    col_idxs = Int32[]
    weights = Float32[]

    g = graph(layer1)

    _NN = wg.NN
    blocksize = (2*_NN+1)^2
    n_conns = nStates(g)*blocksize
    sizehint!(col_idxs, 2*n_conns)
    sizehint!(row_idxs, 2*n_conns)
    sizehint!(weights, 2*n_conns)

    pre_3tuple = Prealloc(NTuple{3, Int32}, blocksize)
    _fillSparseVecs(layer1, layer2, row_idxs, col_idxs, weights, _NN, wg, pre_3tuple)

    append!(row_idxs, col_idxs)
    append!(col_idxs, @view(row_idxs[1:end÷2]))
    append!(weights, weights)

    return row_idxs, col_idxs, weights
end
export genLayerConnections

"""
Give preallocated vectors for row_idxs, col_idxs, and weights
    fills them with the connections withing a layer
"""
function _fillSparseVecs(layer, row_idxs, col_idxs, weights, NN, topology, wg, pre_3tuple, selftype::ST) where ST
    for col_idx in 1:nStates(layer)
        vert_i, vert_j = idxToCoord(col_idx, glength(layer))
        getConnIdxs!(selftype, col_idx, vert_i, vert_j, glength(layer) , gwidth(layer), NN, pre_3tuple)
        for conn in pre_3tuple
            conn_i, conn_j = conn[1], conn[2]
            dr = dist(vert_i, vert_j, conn_i, conn_j, topology)
            # TODO: Overal coords?
            _,_,z = coords(layer)
            dx = abs(conn_i - vert_i)
            dy = abs(conn_j - vert_j)
            dz = 0
            # TODO: Self weight???
            weight = getWeight(wg, dr, (vert_i+conn_i)/2, (vert_j+conn_j)/2, z, dx, dy, dz)
            
            if weight == 0 || isnan(weight)
                continue
            end

            g_col_idx     = idxLToG(col_idx, layer)
            g_conn_idx    = idxLToG(conn[3], layer)

            push!(row_idxs, g_conn_idx)
            push!(col_idxs, g_col_idx)
            push!(weights, weight)

        end
        reset!(pre_3tuple)
    end
end

"""
Give preallocated vectors for row_idxs, col_idxs, and weights
    fills them with the connections withing between two layers
"""
function _fillSparseVecs(layer1, layer2, row_idxs, col_idxs, weights, NN, wg, pre_3tuple)
    for col_idx in 1:nStates(layer1)
        vert_i, vert_j = idxToCoord(col_idx, layer1)
        lattice2_iterator(layer1, layer2, vert_i, vert_j, NN, pre_3tuple)
        for conn in pre_3tuple
            conn_i, conn_j = conn[1], conn[2]
            dr = dist(vert_i, vert_j, conn_i, conn_j, layer1, layer2)
            _,_,z1 = coords(layer1)
            _,_,z2 = coords(layer2)
            dz = z2 - z1
            z_conn = (z1+z2)/2
            conn_i_layer1, conn_j_layer1 = coordsl2tol1(conn_i, conn_j, layer1, layer2) 
            weight = getWeight(wg, dr, (vert_i+conn_i)/2, (vert_j+conn_j)/2, z_conn, conn_i_layer1-vert_i, conn_j_layer1-vert_j, dz)

            if weight == 0 || isnan(weight)
                continue
            end
        
            g_col_idx     = idxLToG(col_idx, layer1)
            g_row_idx    = idxLToG(conn[3], layer2)

            push!(row_idxs, g_row_idx)
            push!(col_idxs, g_col_idx)
            push!(weights, weight)

        end
        reset!(pre_3tuple)
    end
end


abstract type ConnectionReturnType end
struct View <: ConnectionReturnType end
struct Copy <: ConnectionReturnType end

"""
Remove all connections within layer
"""
# @inline removeConnections(layer::IsingLayer) = removeConnections(layer, View)
function removeConnections(layer::IsingLayer)
    old_rows, old_cols, old_weights  = findnz(adj(graph(layer)))

    filter = Vector{Bool}(undef , length(old_rows))

    for idx in eachindex(old_rows)
        @inbounds filter[idx]= !(old_rows[idx] in graphidxs(layer) && old_cols[idx] in graphidxs(layer))
    end

    return (@view old_rows[filter]), (@view old_cols[filter]), (@view old_weights[filter])
end

function removeConnections(layer1, layer2)
    old_rows, old_cols, old_weights  = findnz(adj(graph(layer1)))

    filter = Vector{Bool}(undef , length(old_rows))

    Threads.@threads for idx in eachindex(old_rows)
        filter[idx]= !( (old_rows[idx] in graphidxs(layer1) && old_cols[idx] in graphidxs(layer2)) || 
            (old_rows[idx] in graphidxs(layer2) && old_cols[idx] in graphidxs(layer1)) )
    end

    return (@view old_rows[filter]), (@view old_cols[filter]), (@view old_weights[filter])
end

"""
Remove all connections within layer and going in and out of layer
"""
function removeConnectionsAll(layer::IsingLayer)
    old_rows, old_cols, old_weights  = findnz(adj(graph(layer)))

    filter = Vector{Bool}(undef , length(old_rows))

    Threads.@threads for idx in eachindex(old_rows)
        filter[idx]= !(old_rows[idx] in graphidxs(layer) || old_cols[idx] in graphidxs(layer))
    end

    return (@view old_rows[filter]), (@view old_cols[filter]), (@view old_weights[filter])
end
export removeConnectionsAll

function remConnection!(adj, i, j)
    deleteval!(adj, i, j)
    deleteval!(adj, j, i)
    return adj
end
export remConnection!

remConnectionDirected(adj, i, j) = deleteval!(adj, i, j)

function connectLayersFull(layer1, layer2)
    n_conns = nStates(layer1)*nStates(layer2)

    rows = Vector{Int32}(undef, 2*n_conns)
    cols = similar(rows)
    weights = Vector{Float32}(undef, length(rows))

    for (jdx, j) in collect(enumerate(graphidxs(layer1)))
    # for (jdx, j) in enumerate(graphidxs(layer1))
        for (idx, i) in enumerate(graphidxs(layer2))
            thread_idx = idx + (jdx-1)*length(graphidxs(layer2))

            rows[thread_idx] = i
            cols[thread_idx] = j
            weights[thread_idx] = 1

            rows[n_conns+thread_idx] = j
            cols[n_conns+thread_idx] = i
            weights[n_conns+thread_idx] = 1
        end
    end

    old_rows, old_cols, old_weights = removeConnections(layer1, layer2)

    append!(rows, old_rows)
    append!(cols, old_cols)
    append!(weights, old_weights)

    return rows, cols, weights
end
export connectLayersFull

function connectLayersFull!(layer1, layer2)
    set_adj!(graph(layer1), connectLayersFull(layer1, layer2))
    connections(layer1)[internal_idx(layer1) => internal_idx(layer2)] = :All
    connections(layer2)[internal_idx(layer2) => internal_idx(layer1)] = :All
end
export connectLayersFull!