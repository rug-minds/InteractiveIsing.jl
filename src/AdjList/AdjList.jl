# abstract type AbstractAdjList{Connections} <: AbstractVector{Connections} end

struct AdjList{Connections} <: AbstractVector{Connections}
    data::Vector{Connections}
end

struct TupleConnections <: AbstractVector{Tuple{Int32,Float32}}
    conns::Vector{Tuple{Int32,Float32}}
end

struct VecsConnections
    idxs::Vector{Int32}
    weights::Vector{Float32}
end

@inline idxs(conns::VecsConnections) = conns.idxs
@inline weights(conns::VecsConnections) = conns.weights


AdjTuple(N) = AdjTuple{Tuple{Int32,Float32}}(Vector{Tuple{Int32,Float32}}(undef, N))

# Base.push!(adj::AdjTuple, idx, val) = push!(adj.data[idx], val)
# Base.getindex(adj::AdjTuple, idx) = adj.data[idx]
# Base.length(adj::AdjTuple) = length(adj.data)
# Base.size(adj::AdjTuple) = size(adj.data)
# Base.eltype(adj::AdjTuple) = eltype(adj.data)
# Base.IteratorSize(adj::AdjTuple) = Base.SizeUnknown()
# Base.IteratorEltype(adj::AdjTuple) = Base.HasEltype()

include("WeightGenerator.jl")
include("SquareAdj.jl")
include("SparseAdj.jl")