initSPAdj!(g) = adj(g, spzeros(nStates(g), nStates(g)))
export initSPAdj!

genLayerConnections(layer::AbstractIsingLayer, wg) = genLayerConnections(layer, wg, nstates(graph(layer)))
"""
Give layer and WeightGenerator
    returns the connections within the layer in row_idxs, col_idxs, and weights
"""
function genLayerConnections(layer::AbstractLayerData{D}, precision, wg::WeightGenerator{F,NN,Symmetric}, nstates) where {D,F,NN,Symmetric}
    return _gen_weightgenerator_layer_connections(layer, precision, wg, nstates, Val(Symmetric))
end

function genLayerConnections(layer::AbstractLayerData{D}, precision, wg::PhysicalWeightGenerator{<:WeightGenerator{F,NN,Symmetric}}, nstates) where {D,F,NN,Symmetric}
    return _gen_weightgenerator_layer_connections(layer, precision, wg, nstates, Val(Symmetric))
end

function _gen_weightgenerator_layer_connections(layer::AbstractLayerData{D}, precision, wg, nstates, ::Val{Symmetric}) where {D,Symmetric}
    row_idxs = Int32[]
    col_idxs = Int32[]
    weights = Float32[]

    _NN = getNN(wg)

    # Either global NN is given, or a tuple of NN for each dimension
    @assert (_NN isa Integer || length(_NN) == D )
    
    _NNt = nothing
    if _NN isa Integer # all dimensions same NN
        _NNt = ntuple(i -> _NN, D)
    else
        _NNt = _NN
    end

    blocksize = Int32(prod(2 .* _NNt .+ 1) - 1)

    n_conns = nstates*blocksize

    sizehint_count = Symmetric ? n_conns ÷ 2 : 2 * n_conns
    sizehint!(col_idxs, sizehint_count)::Vector{Int32}
    sizehint!(row_idxs, sizehint_count)::Vector{Int32}
    sizehint!(weights, sizehint_count)::Vector{Float32}

    topology = top(layer)
    if Symmetric
        _fillSparseVecsSymmetricUnique!(layer, precision, row_idxs, col_idxs, weights, topology, wg)
        return (
            LazyConcatVector(row_idxs, col_idxs),
            LazyConcatVector(col_idxs, row_idxs),
            LazyConcatVector(weights, weights),
        )
    end

    _fillSparseVecs(layer, precision, row_idxs, col_idxs, weights, topology, wg)
    return row_idxs, col_idxs, weights
end

function _fillSparseVecs(layer::AbstractLayerData{D}, precision, row_idxs, col_idxs, weights, topology, wg::WG) where {D,WG}
    NN = getNN(wg, D)
    @assert (NN isa Integer || length(NN) == D)
    NNt = NN isa Integer ? ntuple(_ -> NN, Val(D)) : NN

    pr = parentindices(layer)[1]
    layer_size = size(layer)
    LI = LinearIndices(layer_size)
    ps = whichperiodic(topology)
    translation_invariant = is_translation_invariant(topology)

    n_offsets = prod(2 .* NNt .+ 1) - 1
    offsets = Vector{NTuple{D,Int}}(undef, n_offsets)
    dcs = Vector{DeltaCoordinate{D}}(undef, n_offsets)
    drs = Vector{Float64}(undef, n_offsets)

    ranges = ntuple(i -> (-NNt[i]):NNt[i], Val(D))
    o = 1
    for offset_ci in CartesianIndices(ranges)
        delta_offset = offset_ci.I
        all(iszero, delta_offset) && continue

        offsets[o] = delta_offset

        wrapped_offset = ntuple(Val(D)) do i
            di = delta_offset[i]
            if ps[i]
                halfsize = layer_size[i] >>> 1
                abs(di) > halfsize && (di -= sign(di) * layer_size[i])
            end
            di
        end
        # Let the topology define the metric so non-square lattices get the
        # correct distance for weight-generator shells.
        dcs[o] = DeltaCoordinate(wrapped_offset)
        drs[o] = translation_invariant ? delta_distance(topology, dcs[o]) : NaN
        o += 1
    end

    for col_idx in 1:length(layer)
        c1 = Coordinate(topology, col_idx)
        g_col_idx = pr[col_idx]

        for oi in eachindex(offsets)
            c2 = @inline offset(topology, c1, offsets[oi]...; check = false)
            in(c2, topology) || continue

            dr = translation_invariant ? drs[oi] : dist(topology, c1, c2)
            w = precision(getWeight(wg; dr = dr, c1 = c1, c2 = c2, dc = dcs[oi]))
            (w == 0 || isnan(w)) && continue

            conn_idx = LI[c2]
            g_conn_idx = pr[conn_idx]

            push!(row_idxs, Int32(g_conn_idx))
            push!(col_idxs, Int32(g_col_idx))
            push!(weights, w)
        end
    end
    return nothing
end

"""
    _fillSparseVecsSymmetricUnique!(layer, precision, row_idxs, col_idxs, weights, topology, wg)

Fill only one direction for each intra-layer undirected pair. `genLayerConnections`
mirrors these triplets with `LazyConcatVector` when the `WeightGenerator` is
marked symmetric. The kept direction is `global_col_idx < global_row_idx`, so
random generators are evaluated only once per physical edge.
"""
function _fillSparseVecsSymmetricUnique!(layer::AbstractLayerData{D}, precision, row_idxs, col_idxs, weights, topology, wg::WG) where {D,WG}
    NN = getNN(wg, D)
    @assert (NN isa Integer || length(NN) == D)
    NNt = NN isa Integer ? ntuple(_ -> NN, Val(D)) : NN

    pr = parentindices(layer)[1]
    layer_size = size(layer)
    LI = LinearIndices(layer_size)
    ps = whichperiodic(topology)
    translation_invariant = is_translation_invariant(topology)

    n_offsets = prod(2 .* NNt .+ 1) - 1
    offsets = Vector{NTuple{D,Int}}(undef, n_offsets)
    dcs = Vector{DeltaCoordinate{D}}(undef, n_offsets)
    drs = Vector{Float64}(undef, n_offsets)

    ranges = ntuple(i -> (-NNt[i]):NNt[i], Val(D))
    o = 1
    for offset_ci in CartesianIndices(ranges)
        delta_offset = offset_ci.I
        all(iszero, delta_offset) && continue

        offsets[o] = delta_offset

        wrapped_offset = ntuple(Val(D)) do i
            di = delta_offset[i]
            if ps[i]
                halfsize = layer_size[i] >>> 1
                abs(di) > halfsize && (di -= sign(di) * layer_size[i])
            end
            di
        end
        # Let the topology define the metric so non-square lattices get the
        # correct distance for weight-generator shells.
        dcs[o] = DeltaCoordinate(wrapped_offset)
        drs[o] = translation_invariant ? delta_distance(topology, dcs[o]) : NaN
        o += 1
    end

    for col_idx in 1:length(layer)
        c1 = Coordinate(topology, col_idx)
        g_col_idx = pr[col_idx]

        for oi in eachindex(offsets)
            c2 = @inline offset(topology, c1, offsets[oi]...; check = false)
            in(c2, topology) || continue

            conn_idx = LI[c2]
            g_conn_idx = pr[conn_idx]
            g_col_idx < g_conn_idx || continue

            dr = translation_invariant ? drs[oi] : dist(topology, c1, c2)
            w = precision(getWeight(wg; dr = dr, c1 = c1, c2 = c2, dc = dcs[oi]))
            (w == 0 || isnan(w)) && continue

            push!(row_idxs, Int32(g_conn_idx))
            push!(col_idxs, Int32(g_col_idx))
            push!(weights, w)
        end
    end
    return row_idxs, col_idxs, weights
end

"""
Give layer and WeightGenerator
    returns the connections between the two layers in row_idxs, col_idxs, and weights
"""
function genLayerConnections(layer1::AbstractIsingLayer{T1,D}, layer2::AbstractIsingLayer{T2,D}, wg) where {T1,T2,D}
    row_idxs = Int32[]
    col_idxs = Int32[]
    weights = Float32[]

    n_conns = nStates(layer1) * nStates(layer2)
    sizehint!(col_idxs, 2*n_conns)
    sizehint!(row_idxs, 2*n_conns)
    sizehint!(weights, 2*n_conns)

    _fillSparseVecs(layer1, layer2, row_idxs, col_idxs, weights, wg)

    append!(row_idxs, col_idxs)
    append!(col_idxs, @view(row_idxs[1:end÷2]))
    append!(weights, weights)

    return row_idxs, col_idxs, weights
end
export genLayerConnections

"""
Fill row_idxs, col_idxs, and weights with connections between two layers,
    using woorldcoordinates to measure inter-layer distances in a shared space.
"""
function _fillSparseVecs(layer1::AbstractIsingLayer{T1,D}, layer2::AbstractIsingLayer{T2,D}, row_idxs, col_idxs, weights, wg::WG) where {T1,T2,D,WG}
    top1 = topology(layer1)
    top2 = topology(layer2)
    pr1 = parentindices(layer1)[1]
    pr2 = parentindices(layer2)[1]
    LI1 = LinearIndices(size(layer1))
    LI2 = LinearIndices(size(layer2))

    wc2type = typeof(woorldcoordinate(top2, Coordinate(top2, first(CartesianIndices(size(layer2))))))
    wcoords2 = Vector{wc2type}(undef, nStates(layer2))
    for ci2 in CartesianIndices(size(layer2))
        idx2 = LI2[ci2]
        c2 = Coordinate(top2, ci2)
        wcoords2[idx2] = woorldcoordinate(top2, c2)
    end

    for ci1 in CartesianIndices(size(layer1))
        idx1 = LI1[ci1]
        c1 = Coordinate(top1, ci1)
        wc1 = woorldcoordinate(top1, c1)
        g_col_idx = pr1[idx1]

        for idx2 in eachindex(wcoords2)
            wc2 = wcoords2[idx2]
            dr = dist(wc1, wc2)
            w = Float32(getWeight(wg; dr = dr, c1 = wc1, c2 = wc2))
            (w == 0 || isnan(w)) && continue

            push!(row_idxs, Int32(pr2[idx2]))
            push!(col_idxs, Int32(g_col_idx))
            push!(weights, w)
        end
    end
    return nothing
end

"""
Remove all connections within layer
"""
function removeConnections(layer::AbstractIsingLayer)
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
function removeConnectionsAll(layer::AbstractIsingLayer)
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
