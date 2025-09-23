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
function genLayerConnections(layer::AbstractIsingLayer{T,D}, wg) where {T,D}
    row_idxs = Int32[]
    col_idxs = Int32[]
    weights = Float32[]

    g = graph(layer)

    _NN = getNN(wg)
    println("WG: ", wg)
    println("D: ", D)
    
    
    # Either global NN is given, or a tuple of NN for each dimension
    @assert (!(_NN isa Integer) || length(_NN) == D )
    
    blocksize = Int32(0)
    if _NN isa Int32
        blocksize = (2*_NN+1)^D - 1
    else
        blocksize = prod(2 .* _NN .+ 1) - 1
    end

    n_conns = nStates(g)*blocksize
    sizehint!(col_idxs, 2*n_conns)
    sizehint!(row_idxs, 2*n_conns)
    sizehint!(weights, 2*n_conns)

    conn_is = Prealloc(Int32, blocksize)
    conn_js = Prealloc(Int32, blocksize)

    conns = (conn_is, conn_js)
    if D == 3
        conn_ks = Prealloc(Int32, blocksize)
        conns = (conns..., conn_ks)
    end

    conn_idxs = Prealloc(Int32, blocksize)

    _fillSparseVecs(layer, row_idxs, col_idxs, weights, top(layer), wg, conn_idxs, conns...)


    append!(row_idxs, col_idxs)
    append!(col_idxs, @view(row_idxs[1:end÷2]))
    append!(weights, weights)

    return row_idxs, col_idxs, weights
end



"""
From a generator that returns (i, j, idx) for the connections
    fill the row_idxs, col_idxs, and weights
""" 
function _fillSparseVecs(layer::AbstractIsingLayer{T,2}, row_idxs::Vector, col_idxs, weights, topology, @specialize(wg), conns...) where {T}
    NN = wg.NN
    @assert (NN isa Integer || length(NN) == 2)


    conn_idxs, conn_is, conn_js = conns

    NNi = Int32(0)
    NNj = Int32(0)
    if NN isa Integer
        NNi = NN
        NNj = NN
    else
        NNi, NNj = NN
    end

    for col_idx in Int32(1):nstates(layer)
        vert_i, vert_j = idxToCoord(col_idx, glength(layer))
        getConnIdxs!(topology, col_idx, vert_i, vert_j, size(layer), NNi, NNj, conn_idxs, conn_is, conn_js)
        
        @fastmath for conn_num in eachindex(conn_is)
            conn_i = conn_is[conn_num]
            conn_j = conn_js[conn_num]
            conn_idx = conn_idxs[conn_num]

            dr = dist(topology, vert_i, vert_j, conn_i, conn_j)
            # TODO: Overal coords?
            # _,_,z = coords(layer)
            z = 1

            dx, dy = dxdy(topology, (vert_i, vert_j), (conn_i, conn_j))
            # dz = 0
            # x = (vert_i+conn_i)/2
            # y = (vert_j+conn_j)/2
            c1 = Coordinate(vert_i, vert_j)
            c2 = Coordinate(conn_i, conn_j)
            d = c2 - c1

            # weight = getWeight(wg; dr, dx, dy, dz, x, y, z)
            weight = T(wg(;d, c1, c2))

            if !(weight == 0 || isnan(weight))
                g_col_idx     = idxLToG(col_idx, layer)
                g_conn_idx    = idxLToG(conn_idx, layer)

                push!(row_idxs, g_conn_idx)
                push!(col_idxs, g_col_idx)
                push!(weights, weight)
            end

            reset!(conn_is)
            reset!(conn_js)
            reset!(conn_idxs)
        end
        

        if SelfType(wg) == Self()
            weight = getSelfWeight(wg, y = vert_i, x = vert_j;z)
            push!(row_idxs, col_idx)
            push!(col_idxs, col_idx)
            push!(weights, weight)
        end
    end
end

function _fillSparseVecs(layer::AbstractIsingLayer{T,3}, row_idxs, col_idxs, weights, topology, @specialize(wg), conns...) where {T}
    NN = getNN(wg)
    println("Typeof NN: ", typeof(NN))
    @assert (NN isa Integer || length(NN) == 3)

    conn_idxs, conn_is, conn_js, conn_ks = conns
    
    NNi = Int32(0)
    NNj = Int32(0)
    NNk = Int32(0)

    if NN isa Integer
        NNi = NNj = NNk = NN
    else
        NNi, NNj, NNk = NN
    end

    for col_idx in Int32(1):nstates(layer)
        coords_vert = idxToCoord(col_idx, size(layer)) # Coords of the spin
        getConnIdxs!(topology, col_idx, coords_vert, size(layer), NNi, NNj, NNk, conn_idxs, conn_is, conn_js, conn_ks)
        
        @fastmath for conn_num in eachindex(conn_is)
            conn_i = conn_is[conn_num]
            conn_j = conn_js[conn_num]
            conn_k = conn_ks[conn_num]
            conn_idx = conn_idxs[conn_num]

            dr = dist(topology, coords_vert, (conn_i, conn_j, conn_k))
            # TODO: Overal coords?
            # _,_,z = coords(layer)
            dx, dy, dz = dxdydz(topology, coords_vert, (conn_i, conn_j, conn_k))

            vert_i, vert_j, vert_k = coords_vert
            # This doesn't make sense for periodic?
            # x = (vert_i+conn_i)/2
            # y = (vert_j+conn_j)/2
            # z = (vert_k+conn_k)/2
            c1 = Coordinate(vert_i, vert_j, vert_k)
            c2 = Coordinate(conn_i, conn_j, conn_k)
            d = c2 - c1

            weight = T(wg(;d, c1, c2))

            if !(weight == 0 || isnan(weight))
                g_col_idx     = idxLToG(col_idx, layer)
                g_conn_idx    = idxLToG(conn_idx, layer)

                push!(row_idxs, g_conn_idx)
                push!(col_idxs, g_col_idx)
                push!(weights, weight)
            end

            reset!(conn_is)
            reset!(conn_js)
            reset!(conn_ks)
            reset!(conn_idxs)
        end
        

        if SelfType(wg) == Self()
            weight = getSelfWeight(wg, y = vert_i, x = vert_j;z)
            push!(row_idxs, col_idx)
            push!(col_idxs, col_idx)
            push!(weights, weight)
        end
    end
end


"""
Give layer and WeightGenerator
    returns the connections within the layer in row_idxs, col_idxs, and weights
"""
function genLayerConnections(layer1::AbstractIsingLayer{T1,2}, layer2::AbstractIsingLayer{T2,2}, wg) where {T1,T2}
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
    _fillSparseVecs(layer1, layer2, row_idxs, col_idxs, weights, wg, pre_3tuple)

    append!(row_idxs, col_idxs)
    append!(col_idxs, @view(row_idxs[1:end÷2]))
    append!(weights, weights)

    return row_idxs, col_idxs, weights
end
export genLayerConnections


"""
Give preallocated vectors for row_idxs, col_idxs, and weights
    fills them with the connections withing between two layers
"""
function _fillSparseVecs(layer1::IsingLayer, layer2::IsingLayer, row_idxs, col_idxs, weights, wg, pre_3tuple)
    NN = wg.NN

    local NNi = NN
    local NNj = NN
    if NN isa Tuple
        NNi, NNj = NN
    else
        NNi = NN
        NNj = NN
    end

    for col_idx in 1:nStates(layer1)
        vert_i, vert_j = idxToCoord(col_idx, layer1)
        lattice2_iterator(layer1, layer2, vert_i, vert_j, NNi, NNj, pre_3tuple)
        for conn in pre_3tuple
            conn_i, conn_j = conn[1], conn[2]

            dr = dist(layer1, layer2, vert_i, vert_j, conn_i, conn_j)

            _,_,z1 = coords(layer1)
            _,_,z2 = coords(layer2)

            # Project the spin onto layer 1
            conn_i_layer1, conn_j_layer1 = coordsl2tol1(layer1, layer2, conn_i, conn_j)

            #Relative distances between spins
            dx = conn_i_layer1-vert_i
            dy = conn_j_layer1-vert_j
            dz = z2 - z1

            # x,y,z of the connection
            x = (vert_i+conn_i)/2
            y = (vert_j+conn_j)/2
            z = (z1+z2)/2

            weight = getWeight(wg; dr, dx, dy, dz, x, y, z)

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