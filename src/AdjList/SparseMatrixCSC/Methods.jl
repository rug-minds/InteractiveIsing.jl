"""
    weighted_neighbors_sum(node_idx, adj::SparseMatrixCSC, nodevals; transform = identity, transform_weight = identity)

Return the weighted sum over the non-diagonal entries in column `node_idx` of
the sparse adjacency matrix. The row index of each stored value selects the
neighbor value from `nodevals`.
"""
@inline function weighted_neighbors_sum(
    node_idx::I,
    adj::SP,
    nodevals::AV;
    transform::F = identity,
    transform_weight::FW = identity,
) where {I <: Integer, SP <: SparseMatrixCSC, AV <: AbstractArray, F, FW}
    total = @inline column_contraction(node_idx, nodevals, adj; transform, transform_weight)

    # Plain CSC matrices store self-couplings on the diagonal, while this
    # neighbor loop intentionally excludes the node itself.
    self_weight = @inline adj[node_idx, node_idx]
    self_value = @inline getindex(nodevals, node_idx)
    total -= @inline transform_weight(self_weight) * transform(self_value)
    return total
end

"""
    weighted_self(node_idx, adj::SparseMatrixCSC, nodevals; transform = identity, transform_weight = identity)

Return the weighted contribution of the diagonal entry at `node_idx`.
"""
@inline function weighted_self(
    node_idx::I,
    adj::SP,
    nodevals::AV;
    transform::F = identity,
    transform_weight::FW = identity,
) where {I <: Integer, SP <: SparseMatrixCSC, AV <: AbstractArray, F, FW}
    self_weight = @inline adj[node_idx, node_idx]
    self_value = @inline getindex(nodevals, node_idx)
    return transform_weight(self_weight) * transform(self_value)
end
