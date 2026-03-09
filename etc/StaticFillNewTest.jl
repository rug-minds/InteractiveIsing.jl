using SparseArrays
using LoopVectorization
using BenchmarkTools
using JLD2
using LayoutPointers
using LayoutPointers: StaticInt
import VectorizationBase
using VectorizationBase: AbstractSIMD, VecUnroll, vload

"""
    StaticFill{Val,T,N} <: AbstractArray{T,N}

Array-like container that returns a compile-time constant `Val` for every index.
"""
struct StaticFill{Val,T,N} <: AbstractArray{T,N}
    size::NTuple{N,Int}

    function StaticFill(val::T, size::Int...) where {T}
        new{val,T,length(size)}(size)
    end
end

Base.size(sf::StaticFill) = sf.size
Base.length(sf::StaticFill) = prod(sf.size)
Base.IndexStyle(::Type{<:StaticFill}) = IndexLinear()

@inline function Base.getindex(sf::StaticFill{Val,T,N}, idx::Integer) where {Val,T,N}
    @boundscheck checkbounds(sf, idx)
    return Val::T
end

@inline function Base.getindex(sf::StaticFill{Val,T,N}, idxs::Integer...) where {Val,T,N}
    @boundscheck checkbounds(sf, idxs...)
    return Val::T
end

@inline Base.setindex!(::StaticFill, val, idx...) =
    throw(ArgumentError("Cannot set value of a StaticFill, it is immutable"))

LoopVectorization.check_args(::StaticFill) = true

@inline function LayoutPointers.stridedpointer_preserve(sf::StaticFill{V,T,N}) where {V,T,N}
    LayoutPointers.FastRange{T}(T(V), StaticInt{0}()), nothing
end

# ------------------------- deferred_getindex variants -------------------------

# Baseline deferred call (opaque cost in LV)
@inline deferred_getindex_plain(sf::StaticFill{V,T}, j::Integer) where {V,T} = T(V)
@inline deferred_getindex_plain(sf::StaticFill{V,T}, j::AbstractSIMD) where {V,T} = T(V)
@inline deferred_getindex_plain(sf::StaticFill{V,T}, j::VecUnroll) where {V,T} = T(V)
@inline deferred_getindex_plain(sf::StaticFill{V,T}, j) where {V,T} = T(V)

@inline deferred_getindex_plain(v::AbstractVector, j::Integer) = @inbounds v[j]
@inline deferred_getindex_plain(v::AbstractVector, j::AbstractSIMD) = vload(LayoutPointers.stridedpointer(v), (j,))
@inline deferred_getindex_plain(v::AbstractVector, j::VecUnroll) = vload(LayoutPointers.stridedpointer(v), (j,))

LoopVectorization.can_turbo(::typeof(deferred_getindex_plain), ::Val{2}) = true

# Tuned deferred call (same semantics; custom cheap cost tag)
@inline deferred_getindex_tuned(sf::StaticFill{V,T}, j::Integer) where {V,T} = T(V)
@inline deferred_getindex_tuned(sf::StaticFill{V,T}, j::AbstractSIMD) where {V,T} = T(V)
@inline deferred_getindex_tuned(sf::StaticFill{V,T}, j::VecUnroll) where {V,T} = T(V)
@inline deferred_getindex_tuned(sf::StaticFill{V,T}, j) where {V,T} = T(V)

@inline deferred_getindex_tuned(v::AbstractVector, j::Integer) = @inbounds v[j]
@inline deferred_getindex_tuned(v::AbstractVector, j::AbstractSIMD) = vload(LayoutPointers.stridedpointer(v), (j,))
@inline deferred_getindex_tuned(v::AbstractVector, j::VecUnroll) = vload(LayoutPointers.stridedpointer(v), (j,))

LoopVectorization.can_turbo(::typeof(deferred_getindex_tuned), ::Val{2}) = true

const _LV = LoopVectorization

function _LV.instruction!(ls::_LV.LoopSet, f::typeof(deferred_getindex_tuned))
    instr = _LV.gensym!(ls, "f")
    _LV.pushpreamble!(ls, Expr(:(=), instr, f))
    _LV.Instruction(:CHEAPCOMPUTE, instr)
end

function _LV.instruction_cost(instr::_LV.Instruction)
    if instr.mod === :LoopVectorization
        return _LV.COST[instr.instr]
    elseif instr.mod === :CHEAPCOMPUTE
        return _LV.InstructionCost(-3.0, 0.5, 3, 0)
    else
        return _LV.OPAQUE_INSTRUCTION
    end
end

# ------------------------------ benchmark kernels -----------------------------

function col_contraction_native(v::AbstractVector{T}, sp::SparseMatrixCSC{T}, i::Int) where {T}
    tot = zero(T)
    @turbo for ptr in nzrange(sp, i)
        j = sp.rowval[ptr]
        wij = sp.nzval[ptr]
        tot += wij * v[j]
    end
    return tot
end

function col_contraction_deferred_plain(v, sp::SparseMatrixCSC{T}, i::Int) where {T}
    tot = zero(T)
    @turbo for ptr in nzrange(sp, i)
        j = sp.rowval[ptr]
        wij = sp.nzval[ptr]
        tot += wij * deferred_getindex_plain(v, j)
    end
    return tot
end

function col_contraction_deferred_tuned(v, sp::SparseMatrixCSC{T}, i::Int) where {T}
    tot = zero(T)
    @turbo for ptr in nzrange(sp, i)
        j = sp.rowval[ptr]
        wij = sp.nzval[ptr]
        tot += wij * deferred_getindex_tuned(v, j)
    end
    return tot
end

# Same deferred_getindex API, but split on type so StaticFill path does not
# materialize row indices that are semantically unused.
function col_contraction_deferred_split(v::AbstractVector{T}, sp::SparseMatrixCSC{T}, i::Int) where {T}
    tot = zero(T)
    @turbo for ptr in nzrange(sp, i)
        j = sp.rowval[ptr]
        wij = sp.nzval[ptr]
        tot += wij * deferred_getindex_tuned(v, j)
    end
    return tot
end

function col_contraction_deferred_split(v::StaticFill{V,T}, sp::SparseMatrixCSC{T}, i::Int) where {V,T}
    tot = zero(T)
    @turbo for ptr in nzrange(sp, i)
        wij = sp.nzval[ptr]
        tot += wij * deferred_getindex_tuned(v, 1)
    end
    return tot
end

# Type-parameterized baseline requested by user: constant comes from `where {V}`
function col_contraction_typeparam(::Val{V}, sp::SparseMatrixCSC{T}, i::Int) where {V,T}
    c = T(V)
    tot = zero(T)
    @turbo for ptr in nzrange(sp, i)
        j = sp.rowval[ptr]
        wij = sp.nzval[ptr]
        tot += wij * c
    end
    return tot
end

function col_contraction_literal(sp::SparseMatrixCSC{T}, i::Int) where {T}
    tot = zero(T)
    @turbo for ptr in nzrange(sp, i)
        j = sp.rowval[ptr]
        wij = sp.nzval[ptr]
        tot += wij * one(T)
    end
    return tot
end

function first_nonempty_col(sp::SparseMatrixCSC)
    for c in 1:size(sp, 2)
        if !isempty(nzrange(sp, c))
            return c
        end
    end
    return 1
end

function load_matrix()
    if isfile("sparse_50k.jld2")
        A = JLD2.load("sparse_50k.jld2", "A")
        return SparseMatrixCSC(A)
    end

    # Fallback for environments without the pre-generated file.
    return sprand(Float64, 10_000, 10_000, 2e-4)
end

function run_benchmarks()
    sp = load_matrix()
    T = eltype(sp)
    v = rand(T, size(sp, 2))
    sf = StaticFill(one(T), size(sp, 2))
    col = first_nonempty_col(sp)

    println("=== Setup ===")
    println("size(sp)      = ", size(sp))
    println("nnz(sp)       = ", nnz(sp))
    println("column used   = ", col)
    println("nz in column  = ", length(nzrange(sp, col)))

    # Warmup
    col_contraction_native(v, sp, col)
    col_contraction_deferred_plain(v, sp, col)
    col_contraction_deferred_tuned(v, sp, col)
    col_contraction_deferred_plain(sf, sp, col)
    col_contraction_deferred_tuned(sf, sp, col)
    col_contraction_deferred_split(v, sp, col)
    col_contraction_deferred_split(sf, sp, col)
    col_contraction_typeparam(Val(one(T)), sp, col)
    col_contraction_literal(sp, col)

    # Correctness
    println("\n=== Correctness ===")
    r_native_vec = col_contraction_native(v, sp, col)
    r_defer_vec_plain = col_contraction_deferred_plain(v, sp, col)
    r_defer_vec_tuned = col_contraction_deferred_tuned(v, sp, col)
    r_defer_sf_plain = col_contraction_deferred_plain(sf, sp, col)
    r_defer_sf_tuned = col_contraction_deferred_tuned(sf, sp, col)
    r_defer_sf_split = col_contraction_deferred_split(sf, sp, col)
    r_typeparam = col_contraction_typeparam(Val(one(T)), sp, col)
    r_literal = col_contraction_literal(sp, col)

    println("native(Vector)        = ", r_native_vec)
    println("deferred_plain(Vector)= ", r_defer_vec_plain, "  match native = ", isapprox(r_native_vec, r_defer_vec_plain))
    println("deferred_tuned(Vector)= ", r_defer_vec_tuned, "  match native = ", isapprox(r_native_vec, r_defer_vec_tuned))
    println("deferred_plain(SF)    = ", r_defer_sf_plain)
    println("deferred_tuned(SF)    = ", r_defer_sf_tuned)
    println("deferred_split(SF)    = ", r_defer_sf_split)
    println("typeparam(Val{1})     = ", r_typeparam)
    println("literal one(T)        = ", r_literal)
    println("SF tuned ~ typeparam  = ", isapprox(r_defer_sf_tuned, r_typeparam))

    # Benchmarks
    println("\n=== Benchmarks (column contraction) ===")
    print("native v[j] (Vector):        ")
    b_native_vec = @benchmark col_contraction_native($v, $sp, $col)
    display(b_native_vec)

    print("deferred plain (Vector):     ")
    b_defer_vec_plain = @benchmark col_contraction_deferred_plain($v, $sp, $col)
    display(b_defer_vec_plain)

    print("deferred tuned (Vector):     ")
    b_defer_vec_tuned = @benchmark col_contraction_deferred_tuned($v, $sp, $col)
    display(b_defer_vec_tuned)

    print("deferred plain (StaticFill): ")
    b_defer_sf_plain = @benchmark col_contraction_deferred_plain($sf, $sp, $col)
    display(b_defer_sf_plain)

    print("deferred tuned (StaticFill): ")
    b_defer_sf_tuned = @benchmark col_contraction_deferred_tuned($sf, $sp, $col)
    display(b_defer_sf_tuned)

    print("deferred split (Vector):     ")
    b_defer_split_vec = @benchmark col_contraction_deferred_split($v, $sp, $col)
    display(b_defer_split_vec)

    print("deferred split (StaticFill): ")
    b_defer_split_sf = @benchmark col_contraction_deferred_split($sf, $sp, $col)
    display(b_defer_split_sf)

    print("typeparam Val baseline:      ")
    b_typeparam = @benchmark col_contraction_typeparam($(Val(one(T))), $sp, $col)
    display(b_typeparam)

    print("literal baseline:            ")
    b_literal = @benchmark col_contraction_literal($sp, $col)
    display(b_literal)

    t_defer = BenchmarkTools.median(b_defer_sf_tuned).time
    t_defer_split = BenchmarkTools.median(b_defer_split_sf).time
    t_val = BenchmarkTools.median(b_typeparam).time
    ratio = t_defer / t_val
    ratio_split = t_defer_split / t_val
    println("\nMedian time ratio tuned_deferred_SF / typeparam = ", round(ratio; digits=3), "x")
    println("Median time ratio split_deferred_SF / typeparam = ", round(ratio_split; digits=3), "x")
end

run_benchmarks()
