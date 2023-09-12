# NOT USED

struct Connections
    idxs::Vector{Int32}
    weights::Vector{Float32}

    Connections() = new(Int32[], Float32[])
end
@inline push!(conn::Connections, idx, weight) = (push!(conn.idxs, idx); push!(conn.weights, weight))
@inline @inbounds cweight(conns::Connections, idx) = conns.weights[idx]
@inline @inbounds cindex(conns::Connections, idx) = conns.idxs[idx]
@inline Base.eachindex(conns::Connections) = Base.eachindex(conns.idxs)


struct AdjConn{Connections} <: AbstractVector{Connections}
    data::Vector{Connections}
end
export AdjConn

@inline Base.size(adj::AdjConn) = size(adj.data)
@inline Base.length(adj::AdjConn) = length(adj.data)
@inline @inbounds getindex(adj::AdjConn, idx) = adj.data[idx]

function AdjConn(n::Integer)
    data = [Connections() for i in 1:n]
    return AdjConn(data)
end

function adjToAdjConn(adj::Vector)
    n = length(adj)
    adjlist = AdjConn(n)
    for idx in 1:n
        idxs_weights = adj[idx]
        for idx_weight in idxs_weights
            push!(adjlist[idx], idx_weight[1], idx_weight[2])
        end
    end
    
    return adjlist
end
export adjToAdjConn