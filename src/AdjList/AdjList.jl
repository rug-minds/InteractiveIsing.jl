abstract type AdjList{T} <: AbstractVector{T} end

struct AdjTuple{T} <: AdjList{T}
    data::Vector{T}
end

AdjTuple(N) = AdjTuple{Tuple{Int32,Float32}}(Vector{Tuple{Int32,Float32}}(undef, N))

Base.push!(adj::AdjTuple, idx, val) = push!(adj.data[idx], val)
Base.getindex(adj::AdjTuple, idx) = adj.data[idx]
Base.length(adj::AdjTuple) = length(adj.data)
Base.size(adj::AdjTuple) = size(adj.data)
Base.eltype(adj::AdjTuple) = eltype(adj.data)
Base.IteratorSize(adj::AdjTuple) = Base.SizeUnknown()
Base.IteratorEltype(adj::AdjTuple) = Base.HasEltype()

include("SquareAdj.jl")