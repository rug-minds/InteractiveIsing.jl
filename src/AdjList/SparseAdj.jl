initSPAdj!(g) = sp_adj(g, spzeros(nStates(g), nStates(g)))
export initSPAdj!

struct SparseConnections
    row_idxs::Vector{Int32}
    col_idxs::Vector{Int32}
    weights::Vector{Float32}
end

function addConnections(layer, vert_idxs, conn_idxs, weights, replace_rows = nothing, replace_cols = nothing)
    # Either replace rows and replace cols is nothing or both are not nothing
    g = graph(layer)
    _sp_adj = sp_adj(graph(layer))


    removeConnections!(layer, replace_rows, replace_cols)

    old_rows, old_cols, old_weights  = findnz(_sp_adj)
    
    append!(old_rows, vert_idxs, conn_idxs)
    append!(old_cols, conn_idxs, vert_idxs)
    append!(old_weights, weights, weights)

    return sparse(old_rows, old_cols, old_weights, nStates(g), nStates(g))
end
addConnections!(layer, vert_idxs, conn_idxs, weights, replace_rows = nothing, replace_cols = nothing) = sp_adj(graph(layer), addConnections(layer, vert_idxs, conn_idxs, weights, replace_rows, replace_cols))
export addConnections, addConnections!


function coordinates2Cartesian(is,js, ::Val{N} = Val{2}()) where N
    @assert length(is) == length(js)
    return [CartesianIndex{N}(is[idx],js[idx]) for idx in eachindex(is)]
end

Base.@propagate_inbounds function genSPAdj(layer, wg)
    g = graph(layer)

    row_idxs, col_idxs, weights = genLayerConnections(layer, wg)
    old_row_idxs, old_col_idxs, old_weights = removeConnections(layer)

    append!(row_idxs,    old_row_idxs)
    append!(col_idxs,    old_col_idxs)
    append!(weights ,    old_weights)

    return row_idxs, col_idxs, weights
end

@inline genSPAdj!(layer, wg) = set_sp_adj!(graph(layer), genSPAdj(layer, wg))
export genSPAdj!

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
    @time _fillSparseVecs(layer1, layer2, row_idxs, col_idxs, weights, _NN, wg, pre_3tuple)

    append!(row_idxs, col_idxs)
    append!(col_idxs, @view(row_idxs[1:end÷2]))
    append!(weights, weights)

    return row_idxs, col_idxs, weights
end
export genLayerConnections

# For connecting a layer internally
function _fillSparseVecs(layer, row_idxs, col_idxs, weights, NN, topology, wg, pre_3tuple, selftype::ST) where ST
    for col_idx in 1:nStates(layer)
        vert_i, vert_j = idxToCoord(col_idx, glength(layer))
        getConnIdxs!(selftype, col_idx, vert_i, vert_j, glength(layer) , gwidth(layer), NN, pre_3tuple)
        for conn in pre_3tuple
            conn_i, conn_j = conn[1], conn[2]
            dr = dist(vert_i, vert_j, conn_i, conn_j, topology)
            weight = getWeight(wg, dr, (vert_i+conn_i)/2, (vert_j+conn_j)/2)
        
            if weight == 0
                continue
            end

            g_col_idx     = idxLToG(layer, col_idx)
            g_conn_idx    = idxLToG(layer, conn[3])

            push!(row_idxs, g_conn_idx)
            push!(col_idxs, g_col_idx)
            push!(weights, weight)

        end
        reset!(pre_3tuple)
    end
end

# For connecting layers 
function _fillSparseVecs(layer1, layer2, row_idxs, col_idxs, weights, NN, wg, pre_3tuple)
    for col_idx in 1:nStates(layer1)
        vert_i, vert_j = idxToCoord(col_idx, glength(layer1))
        lattice2_iterator(layer1, layer2, vert_i, vert_j, NN, pre_3tuple)
        for conn in pre_3tuple
            conn_i, conn_j = conn[1], conn[2]
            dr = dist(vert_i, vert_j, conn_i, conn_j, layer1, layer2)
            weight = getWeight(wg, dr, (vert_i+conn_i)/2, (vert_j+conn_j)/2)

            if weight == 0
                continue
            end
        
            g_col_idx     = idxLToG(layer1, col_idx)
            g_row_idx    = idxLToG(layer2, conn[3])

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
    old_rows, old_cols, old_weights  = findnz(sp_adj(graph(layer)))

    filter = Vector{Bool}(undef , length(old_rows))

    for idx in eachindex(old_rows)
        @inbounds filter[idx]= !(old_rows[idx] in graphidxs(layer) && old_cols[idx] in graphidxs(layer))
    end

    return (@view old_rows[filter]), (@view old_cols[filter]), (@view old_weights[filter])
end

function removeConnections(layer1, layer2)
    old_rows, old_cols, old_weights  = findnz(sp_adj(graph(layer1)))

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
@inline removeConnectionsAll(layer::IsingLayer) = removeConnectionsAll(layer, View)
function removeConnectionsAll(layer::IsingLayer, ::Type{View})
    # old_rows, old_cols, old_weights  = findnz(sp_adj(graph(layer)))

    # filter = Vector{Bool}(undef , length(old_rows))

    # Threads.@threads for idx in eachindex(old_rows)
    #     filter[idx]= !(old_rows[idx] in graphidxs(layer) || old_cols[idx] in graphidxs(layer))
    # end

    # return (@view old_rows[filter]), (@view old_cols[filter]), (@view old_weights[filter])
    _sp_adj = copy(sp_adj(graph(layer)))
    # for col_idx in graphidxs(layer)
    col_idx = 1
    row_idx_range = 1:10000
        # row_idx_range = _sp_adj.colptr[col_idx]:(_sp_adj.colptr[col_idx+1]-1)
        for row_idx in _sp_adj.rowval[row_idx_range]
            # println("Row idx: ", row_idx, " Col idx: ", col_idx)
            _sp_adj = remConnection!(_sp_adj, row_idx, col_idx)
        end
    # end
    return findnz(_sp_adj)
end

function removeConnectionsAll(layer::IsingLayer, ::Type{Copy})
    old_rows, old_cols, old_weights  = findnz(sp_adj(graph(layer)))

    filter = Vector{Bool}(undef , length(old_rows))

    Threads.@threads for idx in eachindex(old_rows)
        filter[idx]= !(old_rows[idx] in graphidxs(layer) || old_cols[idx] in graphidxs(layer))
    end

    return old_rows[filter], old_cols[filter], old_weights[filter]
end
export removeConnectionsAll

function remConnection!(sp_adj, i, j)
    deleteval!(sp_adj, i, j)
    deleteval!(sp_adj, j, i)
    return sp_adj
end
export remConnection!

