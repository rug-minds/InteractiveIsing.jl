initSPAdj!(g) = sp_adj(g, spzeros(nStates(g), nStates(g)))
export initSPAdj!

struct SparseConnections
    row_idxs::Vector{Int32}
    col_idxs::Vector{Int32}
    weights::Vector{Float32}
end

Base.@propagate_inbounds function genSPAdj(layer, wg)
    row_idxs = Int32[]
    col_idxs = Int32[]
    weights = Float32[]

    g = graph(layer)

    _NN = wg.NN
    blocksize = (2*_NN+1)^2
    n_conns = nStates(g)*blocksize
    sizehint!(col_idxs, n_conns)
    sizehint!(row_idxs, n_conns)
    sizehint!(weights, n_conns)

    pre_3tuple = SPrealloc(NTuple{3, Int32}, 2*blocksize)
    _fillSparseVecs(layer, row_idxs, col_idxs, weights, _NN, top(layer), wg, pre_3tuple, SelfType(wg))

    addConnections(layer, row_idxs, col_idxs, weights, :All)
end
@inline genSPAdj!(layer, wg) = sp_adj(graph(layer), genSPAdj(layer, wg))
export genSPAdj, genSPAdj!

function _fillSparseVecs(layer, row_idxs, col_idxs, weights, NN, topology, wg, pre_3tuple, selftype::ST) where ST
    for row_idx in 1:nStates(layer)
        vert_i, vert_j = idxToCoord(row_idx, glength(layer))
        getConnIdxs!(selftype, row_idx, vert_i, vert_j, glength(layer) , gwidth(layer), NN, pre_3tuple)
        for conn in pre_3tuple
            conn_i, conn_j = conn[1], conn[2]
            dr = dist(vert_i, vert_j, conn_i, conn_j, topology)
            weight = getWeight(wg, dr, (vert_i+conn_i)/2, (vert_j+conn_j)/2)
        
            conn_idx = conn[3]

            push!(row_idxs, row_idx)
            push!(col_idxs, conn_idx)
            push!(weights, weight)

        end
        reset!(pre_3tuple)
    end
end

function addConnections(layer, vert_idxs, conn_idxs, weights, replace_rows = nothing, replace_cols = nothing)
    # Either replace rows and replace cols is nothing or both are not nothing
    g = graph(layer)
    _sp_adj = sp_adj(graph(layer))


    removeConnections!(layer, replace_rows, replace_cols)

    old_rows, old_cols, old_weights  = findnz(_sp_adj)
    
    append!(old_rows, vert_idxs)
    append!(old_cols, conn_idxs)
    append!(old_weights, weights)

    append!(old_rows, conn_idxs)
    append!(old_cols, vert_idxs)
    append!(old_weights, weights)

    return sparse(old_rows, old_cols, old_weights, nStates(g), nStates(g))
end
addConnections!(layer, vert_idxs, conn_idxs, weights, replace_rows = nothing, replace_cols = nothing) = sp_adj(graph(layer), addConnections(layer, vert_idxs, conn_idxs, weights, replace_rows, replace_cols))
export addConnections, addConnections!

export removeConnections, removeConnections!

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

@inline genSPAdj!(layer, wg ) = sp_adj(g, sparse(row_idxs, col_idxs, weights, nStates(g), nStates(g)))
export genSPAdj!

function clearLayerConections(layer)

end

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

    pre_3tuple = SPrealloc(NTuple{3, Int32}, 2*blocksize)
    _fillSparseVecs(layer, row_idxs, col_idxs, weights, _NN, top(layer), wg, pre_3tuple, SelfType(wg))

    append!(row_idxs, col_idxs)
    append!(col_idxs, @view(row_idxs[1:end÷2]))
    append!(weights, weights)

    return row_idxs, col_idxs, weights
end

abstract type ConnectionReturnType end
struct View <: ConnectionReturnType end
struct Copy <: ConnectionReturnType end

"""
Remove all connections within layer
"""
@inline removeConnections(layer::IsingLayer) = removeConnections(layer, View)
function removeConnections(layer::IsingLayer, ::Type{View})
    old_rows, old_cols, old_weights  = findnz(sp_adj(graph(layer)))

    filter = Vector{Bool}(undef , length(old_rows))

    Threads.@threads for idx in eachindex(old_rows)
        filter[idx]= !(old_rows[idx] in graphidxs(layer) && old_cols[idx] in graphidxs(layer))
    end

    return (@view old_rows[filter]), (@view old_cols[filter]), (@view old_weights[filter])
end
function removeConnections(layer::IsingLayer, ::Type{Copy})
    old_rows, old_cols, old_weights  = findnz(sp_adj(graph(layer)))

    filter = Vector{Bool}(undef , length(old_rows))

    Threads.@threads for idx in eachindex(old_rows)
        filter[idx]= !(old_rows[idx] in graphidxs(layer) && old_cols[idx] in graphidxs(layer))
    end

    return old_rows[filter], old_cols[filter], old_weights[filter]
end

function removeConnections(layer1, layer2, ::Type{View})
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
    old_rows, old_cols, old_weights  = findnz(sp_adj(graph(layer)))

    filter = Vector{Bool}(undef , length(old_rows))

    Threads.@threads for idx in eachindex(old_rows)
        filter[idx]= !(old_rows[idx] in graphidxs(layer) || old_cols[idx] in graphidxs(layer))
    end

    return (@view old_rows[filter]), (@view old_cols[filter]), (@view old_weights[filter])
end

function removeConnectionsAll(layer::IsingLayer, ::Type{Copy})
    old_rows, old_cols, old_weights  = findnz(sp_adj(graph(layer)))

    filter = Vector{Bool}(undef , length(old_rows))

    Threads.@threads for idx in eachindex(old_rows)
        filter[idx]= !(old_rows[idx] in graphidxs(layer) || old_cols[idx] in graphidxs(layer))
    end

    return old_rows[filter], old_cols[filter], old_weights[filter]
end

function removeConnections(sp_adj, vert_idxs, conn_idxs)
    sp_adj = copy(sp_adj)

    sp_adj[coordinates2Cartesian(vert_idxs, conn_idxs)] .= 0
    sp_adj[coordinates2Cartesian(conn_idxs, vert_idxs)] .= 0

    dropzeros!(sp_adj)

    return findnz(sp_adj)
end
export removeConnections

function remConnection!(sp_adj, i, j)
    deleteval!(sp_adj, i, j)
    deleteval!(sp_adj, j, i)
    return sp_adj
end
export remConnection!

function deleteval!(sp_adj, i,j) 
    searchrange = sp_adj.colptr[j]:(sp_adj.colptr[j+1]-1)
    idx = findfirst(x -> x == i, sp_adj.rowval , searchrange)
    if !isnothing(idx)
        deleteat!(sp_adj.rowval, idx)
        deleteat!(sp_adj.nzval, idx)
        sp_adj.colptr[j+1:end] .-= 1
    end
    return sp_adj
end
export deleteval!

function Base.findfirst(predicate::Function, A, searchrange::UnitRange)
    idx = findfirst(predicate, (@view A[searchrange]))

    !isnothing(idx) && (idx += searchrange.start - 1)
    return idx
end