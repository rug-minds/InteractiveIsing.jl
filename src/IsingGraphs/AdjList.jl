struct Connections
    idxs::Vector{Int32}
    weights::Vector{Float32}

    Connections() = new(Int32[], Float32[])
end
@inline push!(conn::Connections, idx, weight) = (push!(conn.idxs, idx); push!(conn.weights, weight))
@inline @inbounds cweight(conns::Connections, idx) = conns.weights[idx]
@inline @inbounds cindex(conns::Connections, idx) = conns.idxs[idx]
@inline Base.eachindex(conns::Connections) = Base.eachindex(conns.idxs)


struct AdjList{Connections} <: AbstractVector{Connections}
    data::Vector{Connections}
end
export AdjList

@inline Base.size(adj::AdjList) = size(adj.data)
@inline Base.length(adj::AdjList) = length(adj.data)
@inline @inbounds getindex(adj::AdjList, idx) = adj.data[idx]

function AdjList(n::Integer)
    data = [Connections() for i in 1:n]
    return AdjList(data)
end

function adjToAdjList(adj::Vector)
    n = length(adj)
    adjlist = AdjList(n)
    for idx in 1:n
        idxs_weights = adj[idx]
        for idx_weight in idxs_weights
            push!(adjlist[idx], idx_weight[1], idx_weight[2])
        end
    end
    
    return adjlist
end


