using LoopVectorization
using LayoutPointers
using LayoutPointers: StaticInt
using SparseArrays
using BenchmarkTools
using JLD2
import VectorizationBase
using VectorizationBase: AbstractSIMD, VecUnroll, vload

"""
    StaticFill{Val, T, N} <: AbstractArray{T,N}

An AbstractArray whose value is baked into the type parameter `Val`.
Every index returns the same compile-time constant.
"""
struct StaticFill{Val, T, N} <: AbstractArray{T,N}
    size::NTuple{N, Int}

    function StaticFill(val::T, size::Int...) where {T}
        new{val, T, length(size)}(size)
    end
end

Base.size(sf::StaticFill) = sf.size
Base.length(sf::StaticFill) = prod(sf.size)
Base.IndexStyle(::Type{<:StaticFill}) = IndexLinear()

@inline function Base.getindex(sf::StaticFill{Val, T, N}, idx::Integer) where {Val, T, N}
    @boundscheck checkbounds(sf, idx)
    return Val::T
end

@inline function Base.getindex(sf::StaticFill{Val, T, N}, idxs::Integer...) where {Val, T, N}
    @boundscheck checkbounds(sf, idxs...)
    return Val::T
end

@inline Base.setindex!(::StaticFill, val, idx...) =
    throw(ArgumentError("Cannot set value of a StaticFill, it is immutable"))

Base.similar(sf::StaticFill{V,T,N}, ::Type{S}, dims::Dims) where {V,T,N,S} = Array{S}(undef, dims)

Base.iterate(sf::StaticFill{Val,T}, state=1) where {Val,T} =
    state > length(sf) ? nothing : (Val::T, state + 1)

# Direct v[j] path for @turbo.
LoopVectorization.check_args(::StaticFill) = true

@inline function LayoutPointers.stridedpointer_preserve(sf::StaticFill{V,T,N}) where {V,T,N}
    LayoutPointers.FastRange{T}(T(V), StaticInt{0}()), nothing
end

# Compute-op path for @turbo.
@inline deferred_getindex(sf::StaticFill{V,T}, j::Integer) where {V,T} = T(V)
@inline deferred_getindex(sf::StaticFill{V,T}, j::AbstractSIMD) where {V,T} = T(V)
@inline deferred_getindex(sf::StaticFill{V,T}, j::VecUnroll) where {V,T} = T(V)
@inline deferred_getindex(sf::StaticFill{V,T}, j) where {V,T} = T(V)

@inline deferred_getindex(v::AbstractVector, j::Integer) = @inbounds v[j]
@inline deferred_getindex(v::AbstractVector, j::AbstractSIMD) = vload(LayoutPointers.stridedpointer(v), (j,))
@inline deferred_getindex(v::AbstractVector, j::VecUnroll) = vload(LayoutPointers.stridedpointer(v), (j,))

# Backward-compatible alias.
@inline turbo_getindex(args...) = deferred_getindex(args...)

LoopVectorization.can_turbo(::typeof(deferred_getindex), ::Val{2}) = true
LoopVectorization.can_turbo(::typeof(turbo_getindex), ::Val{2}) = true

@inline function column_contraction(v::AbstractVector{T}, sp::SparseMatrixCSC{T}, i) where {T}
    tot = zero(T)
    rowval = SparseArrays.getrowval(sp)
    nzval = SparseArrays.getnzval(sp)
    @turbo for ptr in nzrange(sp, i)
        j = rowval[ptr]
        wij = nzval[ptr]
        v = turbo_getindex(v, j)
        tot += wij * v
    end
    return tot
end

# StaticFill specialization: index is semantically irrelevant, so avoid
# loading `rowval` to match literal/Val-style performance.
@inline function column_contraction(v::StaticFill{V,T}, sp::SparseMatrixCSC{T}, i) where {V,T}
    tot = zero(T)
    @turbo for ptr in nzrange(sp, i)
        wij = sp.nzval[ptr]
        j = sp.rowval[ptr] # Still need to load for boundscheck, but LLVM should DCE it from the multiply.
        tot += wij * turbo_getindex(v, j)
    end
    return tot
end

const m = JLD2.load("sparse_50k.jld2", "A")
const sf = StaticFill(5.0, size(m, 2))
@benchmark column_contraction($sf, $m, 1)
