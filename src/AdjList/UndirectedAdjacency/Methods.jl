@inline function weighted_neighbors_sum(node_idx, adj::UA, nodevals::AV; transform::F = identity, transform_weight::FW = identity) where {UA <: UndirectedAdjacency, AV<:AbstractArray, F, FW}
    total = @inline column_contraction(node_idx, nodevals, adj.sp; transform, transform_weight)
    if !separate_diagonal(adj)
        self_weight = adj.sp[node_idx, node_idx]
        self_value = @inline getindex(nodevals, node_idx)
        total -= @inline transform_weight(self_weight) * transform(self_value)
    end
    return total
end

@inline function weighted_self(node_idx, adj::UA, nodevals::AV; transform::F = identity, transform_weight::FW = identity) where {UA <: UndirectedAdjacency, AV<:AbstractVector, F, FW}
    self_weight = @inline adj[node_idx, node_idx]
    self_value = @inline getindex(nodevals, node_idx)
    return transform_weight(self_weight) * transform(self_value)
end
