using SparseArrays, LoopVectorization, BenchmarkTools, JLD2, LinearAlgebra


"""
    StaticFill{Val, T, N} <: AbstractArray{T,N}

An AbstractArray whose value is baked into the type parameter `Val`.
Every index returns the same compile-time constant. Useful as a drop-in
replacement for `Vector`/`Array` when a parameter is spatially uniform,
enabling the compiler to constant-fold and SIMD-broadcast the value.

Supports `@turbo` (LoopVectorization) via a zero-stride `StridedPointer`:
SIMD loads read from a single backing cell and broadcast across lanes.
"""
struct StaticFill{Val, T, N} <: AbstractArray{T,N}
    backing::Vector{T}      # single-element backing for SIMD loads
    size::NTuple{N, Int}

    function StaticFill(val::T, size::Int...) where T
        new{val, T, length(size)}(T[val], size)
    end
end

# ── Base AbstractArray interface ──────────────────────────────────────────────

Base.size(sf::StaticFill) = sf.size
Base.length(sf::StaticFill) = prod(sf.size)

@inline function Base.getindex(sf::StaticFill{Val, T, N}, idx::Integer) where {Val, T, N}
    @boundscheck checkbounds(sf, idx)
    return Val::T
end

@inline function Base.getindex(sf::StaticFill{Val, T, N}, idxs::Integer...) where {Val, T, N}
    @boundscheck checkbounds(sf, idxs...)
    return Val::T
end

@inline Base.setindex!(::StaticFill, val, idx...) = throw(ArgumentError("Cannot set value of a StaticFill, it is immutable"))

Base.IndexStyle(::Type{<:StaticFill}) = IndexLinear()
Base.similar(sf::StaticFill{V,T,N}, ::Type{S}, dims::Dims) where {V,T,N,S} = Array{S}(undef, dims)

Base.strides(::StaticFill{V,T,N}) where {V,T,N} = ntuple(_ -> 0, N)
@inline Base.pointer(sf::StaticFill{V,T}) where {V,T} = pointer(sf.backing)

Base.iterate(sf::StaticFill{Val,T}, state=1) where {Val,T} = state > length(sf) ? nothing : (Val::T, state + 1)

# ── LoopVectorization / @turbo integration ────────────────────────────────────

using LoopVectorization
using LayoutPointers
using LayoutPointers: StaticInt
const _SAI = LayoutPointers.StaticArrayInterface

LoopVectorization.check_args(::StaticFill) = true
_SAI.dense_dims(::Type{<:StaticFill{V,T,N}}) where {V,T,N} = ntuple(_ -> _SAI.static(true), N)

@inline function LayoutPointers.memory_reference(sf::StaticFill)
    p = pointer(sf.backing)
    return p, sf.backing
end

@inline function LayoutPointers.stridedpointer_preserve(sf::StaticFill{V,T,N}) where {V,T,N}
    p, r = LayoutPointers.memory_reference(sf)
    si = _SAI.StrideIndex{N, ntuple(identity, N), 1}(
        ntuple(_ -> StaticInt{0}(), N),   # zero byte-stride: every index reads same cell
        ntuple(_ -> StaticInt{1}(), N)    # 1-based offsets
    )
    bsi = LayoutPointers.mulstrides(T, si)
    LayoutPointers.stridedpointer(p, bsi, StaticInt{0}()), r
end

function col_contraction(v::AbstractVector{T}, sp::SparseMatrixCSC{T}, i) where {V,T}
    tot = zero(T)
    @turbo for ptr in nzrange(sp, i)
        j = sp.rowval[ptr]
        wij = sp.nzval[ptr]
        # tot += wij * v[j]
        tot += wij * v[j]

    end
    return tot
end

@inline indirect_getindex(v::AbstractVector{T}, idx) where T = v[idx]

function col_contraction(v::StaticFill{V,T}, sp::SparseMatrixCSC{T}, i) where {V,T}
    tot = zero(T)
    @turbo for ptr in nzrange(sp, i)
        j = sp.rowval[ptr]
        wij = sp.nzval[ptr]
        # tot += wij * v[j]
        tot += wij * v[j]

    end
    return tot
end

function col_contraction2(v::AbstractVector{T}, sp::SparseMatrixCSC{T}, i) where {V,T}
    tot = zero(T)
    @turbo for ptr in nzrange(sp, i)
        j = sp.rowval[ptr]
        wij = sp.nzval[ptr]
        # tot += wij * v[j]
        v = indirect_getindex(v, j)
        tot += wij * v

    end
    return tot
end

m = JLD2.load("sparse_50k.jld2", "A")
const i = 1

v = rand(size(m, 2))
sf = StaticFill(1.0, size(m,2))
c = @view m[:,i]


@benchmark col_contraction(sf, m, i)
@benchmark col_contraction(v, m, i)
@benchmark dot((@view m[:,i]), v)
col_contraction2(v, m, i)