using LoopVectorization, LayoutPointers, SparseArrays, JLD2, BenchmarkTools
using LayoutPointers: StaticInt
import VectorizationBase
using VectorizationBase: AbstractSIMD, VecUnroll, vload

struct StaticFill{Val, T, N} <: AbstractArray{T,N}
    size::NTuple{N, Int}
    function StaticFill(val::T, size::Int...) where T
        new{val, T, length(size)}(size)
    end
end
Base.size(sf::StaticFill) = sf.size
Base.length(sf::StaticFill) = prod(sf.size)
Base.IndexStyle(::Type{<:StaticFill}) = IndexLinear()
@inline Base.getindex(sf::StaticFill{Val,T}, idx::Integer) where {Val,T} = Val::T

LoopVectorization.check_args(::StaticFill) = true
@inline function LayoutPointers.stridedpointer_preserve(sf::StaticFill{V,T,N}) where {V,T,N}
    LayoutPointers.FastRange{T}(T(V), StaticInt{0}()), nothing
end

@inline turbo_getindex(sf::StaticFill{V,T}, j::Integer) where {V,T} = T(V)
@inline turbo_getindex(sf::StaticFill{V,T}, j::AbstractSIMD) where {V,T} = T(V)
@inline turbo_getindex(sf::StaticFill{V,T}, j::VecUnroll) where {V,T} = T(V)
@inline turbo_getindex(sf::StaticFill{V,T}, j) where {V,T} = T(V)
@inline turbo_getindex(v::AbstractVector, j::Integer) = @inbounds v[j]
@inline turbo_getindex(v::AbstractVector, j::AbstractSIMD) = @inline vload(LayoutPointers.stridedpointer(v), (j,))
@inline turbo_getindex(v::AbstractVector, j::VecUnroll) = @inline vload(LayoutPointers.stridedpointer(v), (j,))
LoopVectorization.can_turbo(::typeof(turbo_getindex), ::Val{2}) = true

# Default unroll
@inline function col_default(v, sp::SparseMatrixCSC{T}, i) where T
    tot = zero(T)
    @turbo for ptr in nzrange(sp, i)
        j = sp.rowval[ptr]
        wij = sp.nzval[ptr]
        tot += wij * turbo_getindex(v, j)
    end
    return tot
end

# Forced unroll=2
@inline function col_u2(v, sp::SparseMatrixCSC{T}, i) where T
    tot = zero(T)
    @turbo unroll=2 for ptr in nzrange(sp, i)
        j = sp.rowval[ptr]
        wij = sp.nzval[ptr]
        tot += wij * turbo_getindex(v, j)
    end
    return tot
end

# Forced unroll=4
@inline function col_u4(v, sp::SparseMatrixCSC{T}, i) where T
    tot = zero(T)
    @turbo unroll=4 for ptr in nzrange(sp, i)
        j = sp.rowval[ptr]
        wij = sp.nzval[ptr]
        tot += wij * turbo_getindex(v, j)
    end
    return tot
end

# Forced unroll=8
@inline function col_u8(v, sp::SparseMatrixCSC{T}, i) where T
    tot = zero(T)
    @turbo unroll=8 for ptr in nzrange(sp, i)
        j = sp.rowval[ptr]
        wij = sp.nzval[ptr]
        tot += wij * turbo_getindex(v, j)
    end
    return tot
end

# Literal baseline
@inline function col_literal(sp::SparseMatrixCSC{T}, i) where T
    tot = zero(T)
    @turbo for ptr in nzrange(sp, i)
        wij = sp.nzval[ptr]
        tot += wij * 1.0
    end
    return tot
end

m = JLD2.load("sparse_50k.jld2", "A")
sf = StaticFill(1.0, size(m,2))

# warmup
col_default(sf, m, 1); col_u2(sf, m, 1); col_u4(sf, m, 1); col_u8(sf, m, 1); col_literal(m, 1)

println("=== turbo_getindex(StaticFill) with different unroll factors ===")
print("default:   "); show(stdout, MIME"text/plain"(), @benchmark col_default($sf, $m, 1)); println()
print("unroll=2:  "); show(stdout, MIME"text/plain"(), @benchmark col_u2($sf, $m, 1)); println()
print("unroll=4:  "); show(stdout, MIME"text/plain"(), @benchmark col_u4($sf, $m, 1)); println()
print("unroll=8:  "); show(stdout, MIME"text/plain"(), @benchmark col_u8($sf, $m, 1)); println()
print("literal:   "); show(stdout, MIME"text/plain"(), @benchmark col_literal($m, 1)); println()
