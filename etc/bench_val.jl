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
LoopVectorization.can_turbo(::typeof(turbo_getindex), ::Val{2}) = true

# turbo_getindex, default unroll
@inline function col_tg_default(v, sp::SparseMatrixCSC{T}, i) where T
    tot = zero(T)
    @turbo for ptr in nzrange(sp, i)
        j = sp.rowval[ptr]
        wij = sp.nzval[ptr]
        tot += wij * turbo_getindex(v, j)
    end
    return tot
end

# turbo_getindex, unroll=8
@inline function col_tg_u8(v, sp::SparseMatrixCSC{T}, i) where T
    tot = zero(T)
    @turbo unroll=8 for ptr in nzrange(sp, i)
        j = sp.rowval[ptr]
        wij = sp.nzval[ptr]
        tot += wij * turbo_getindex(v, j)
    end
    return tot
end

# literal 1.0
@inline function col_lit1(sp::SparseMatrixCSC{T}, i) where T
    tot = zero(T)
    @turbo for ptr in nzrange(sp, i)
        wij = sp.nzval[ptr]
        tot += wij * 1.0
    end
    return tot
end

# literal 5.0
@inline function col_lit5(sp::SparseMatrixCSC{T}, i) where T
    tot = zero(T)
    @turbo for ptr in nzrange(sp, i)
        wij = sp.nzval[ptr]
        tot += wij * 5.0
    end
    return tot
end

# plain sum (no multiply at all)
@inline function col_sum(sp::SparseMatrixCSC{T}, i) where T
    tot = zero(T)
    @turbo for ptr in nzrange(sp, i)
        wij = sp.nzval[ptr]
        tot += wij
    end
    return tot
end

m = JLD2.load("sparse_50k.jld2", "A")
sf1 = StaticFill(1.0, size(m,2))
sf5 = StaticFill(5.0, size(m,2))

# warmup
col_tg_default(sf1, m, 1); col_tg_u8(sf1, m, 1)
col_tg_default(sf5, m, 1); col_tg_u8(sf5, m, 1)
col_lit1(m, 1); col_lit5(m, 1); col_sum(m, 1)

println("=== val=1.0 ===")
print("tg default:  "); show(stdout, MIME"text/plain"(), @benchmark col_tg_default($sf1, $m, 1)); println()
print("tg unroll=8: "); show(stdout, MIME"text/plain"(), @benchmark col_tg_u8($sf1, $m, 1)); println()
print("literal 1.0: "); show(stdout, MIME"text/plain"(), @benchmark col_lit1($m, 1)); println()

println("\n=== val=5.0 ===")
print("tg default:  "); show(stdout, MIME"text/plain"(), @benchmark col_tg_default($sf5, $m, 1)); println()
print("tg unroll=8: "); show(stdout, MIME"text/plain"(), @benchmark col_tg_u8($sf5, $m, 1)); println()
print("literal 5.0: "); show(stdout, MIME"text/plain"(), @benchmark col_lit5($m, 1)); println()

println("\n=== plain sum (no multiply) ===")
print("sum:         "); show(stdout, MIME"text/plain"(), @benchmark col_sum($m, 1)); println()
