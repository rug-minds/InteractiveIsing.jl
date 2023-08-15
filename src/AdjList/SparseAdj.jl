initSPAdj!(g) = sp_adj(g, spzeros(nStates(g), nStates(g)))
export initSPAdj!

Base.@propagate_inbounds function genSPAdj!(layer, wg)
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
    _fillSparseVecs(layer, col_idxs, row_idxs, weights, _NN, top(layer), wg, pre_3tuple, SelfType(wg))

    remaining_col_idxs = setdiff(1:nStates(g), graphidxs(layer))
    old_rows, old_cols, old_weights  = findnz(sp_adj(g)[remaining_col_idxs, :])

    append!(row_idxs, old_rows)
    append!(col_idxs, old_cols)
    append!(weights, old_weights)

    display(row_idxs)
    display(col_idxs)
    display(weights)

    sp_adj(g, sparse(row_idxs, col_idxs, weights, nStates(g), nStates(g)))

end
export genSPAdj!

function _fillSparseVecs(layer, col_idxs, row_idxs, weights, NN, topology, wg, pre_3tuple, selftype::ST) where ST
    for row_idx in 1:nStates(layer)
        vert_i, vert_j = idxToCoord(row_idx, glength(layer))
        getConnIdxs!(selftype, row_idx, vert_i, vert_j, glength(layer) , gwidth(layer), NN, pre_3tuple)
        for conn in pre_3tuple
            conn_i, conn_j = conn[1], conn[2]
            dr = dist(vert_i, vert_j, conn_i, conn_j, topology)
            weight = getWeight(wg, dr, (vert_i+conn_i)/2, (vert_j+conn_j)/2)
            
            g_row_idx = idxLToG(layer, row_idx)
            g_conn_idx = idxLToG(layer, conn[3])
            
            push!(row_idxs, g_row_idx)
            push!(col_idxs, g_conn_idx)

            push!(row_idxs, g_conn_idx)
            push!(col_idxs, g_row_idx)

            push!(weights, weight)
            push!(weights, weight)

        end
        reset!(pre_3tuple)
    end
end

function addConnections(layer, vert_idxs, conn_idxs, weights, replace_rows = nothing, replace_cols = nothing)
    # Either replace rows and replace cols is nothing or both are not nothing
    @assert isnothing(replace_rows) && isnothing(replace_cols) || !isnothing(replace_rows) && !isnothing(replace_cols)

    sp_adj = sp_adj(graph(layer))

    # If we are replacing connections we need to remove the old ones
    if !isnothing(replace_rows)
        sp_adj[coordinates2Cartesian(vert_idxs, conn_idxs)] .= 0
        sp_adj[coordinates2Cartesian(conn_idxs, vert_idxs)] .= 0
        dropzeros!(sp_adj)
    end

    old_rows, old_cols, old_weights  = findnz(sp_adj(g))
    append!(old_rows, vert_idxs)
    append!(old_cols, conn_idxs)
    append!(old_weights, weights)
    return sp_adj(g, sparse(old_rows, old_cols, old_weights, nStates(g), nStates(g)))
end

function coordinates2Cartesian(is,js, ::Val{N} = Val{2}()) where N
    @assert length(is) == length(js)
    return [CartesianIndex{N}(is[idx],js[idx]) idx in eachindex(is)]
end