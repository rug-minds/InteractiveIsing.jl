# using LoopVectorization
using LayoutPointers
using LayoutPointers: StaticInt
import VectorizationBase
const _SAI_FA = LayoutPointers.StaticArrayInterface


# ══════════════════════════════════════════════════════════════════════════════
# FillArray — mutable single-value array (Ref-backed)
# ══════════════════════════════════════════════════════════════════════════════

"""
    FillArray{T, N} <: AbstractArray{T,N}

An AbstractArray backed by a single `Ref{T}`. Every index returns the same
value, but unlike `StaticFill`, the value can be changed at runtime via `fa[] = newval`.

Supports `@turbo` via zero-stride `StridedPointer` (same mechanism as `StaticFill`).
"""
struct FillArray{T, N} <: AbstractArray{T,N}
    val::Base.RefValue{T}
    size::NTuple{N, Int}
end

FillArray(val::T, size::Int...) where T = FillArray{T, length(size)}(Ref(val), size)

# ── Base AbstractArray interface ──────────────────────────────────────────────

Base.size(fa::FillArray) = fa.size
Base.length(fa::FillArray) = prod(fa.size)

@inline function Base.getindex(fa::FillArray{T}, idx::Integer) where T
    @boundscheck checkbounds(fa, idx)
    return fa.val[]
end

@inline function Base.getindex(fa::FillArray{T}, idxs::Integer...) where T
    @boundscheck checkbounds(fa, idxs...)
    return fa.val[]
end

@inline function Base.setindex!(fa::FillArray{T}, v, idx::Integer) where T
    @boundscheck checkbounds(fa, idx)
    fa.val[] = v
end

@inline function Base.setindex!(fa::FillArray{T}, v) where T
    fa.val[] = v
end

Base.IndexStyle(::Type{<:FillArray}) = IndexLinear()
Base.similar(fa::FillArray{T,N}, ::Type{S}, dims::Dims) where {T,N,S} = Array{S}(undef, dims)

Base.strides(::FillArray{T,N}) where {T,N} = ntuple(_ -> 0, N)
@inline Base.pointer(fa::FillArray{T}) where T = Ptr{T}(pointer_from_objref(fa.val))

Base.iterate(fa::FillArray{T}, state=1) where T = state > length(fa) ? nothing : (fa.val[], state + 1)

# ── LoopVectorization / @turbo integration ────────────────────────────────────

LoopVectorization.check_args(::FillArray) = true
_SAI_FA.dense_dims(::Type{<:FillArray{T,N}}) where {T,N} = ntuple(_ -> _SAI_FA.static(true), N)

# ── turbo_getindex compute-op path ────────────────────────────────────────────
# FillArray returns the same value for every index, so ignore j entirely.
# This avoids going through stridedpointer (which fails for zero-stride arrays)
# and lets LLVM DCE the index chain.
@inline turbo_getindex(fa::FillArray, j::Integer) = fa.val[]
@inline turbo_getindex(fa::FillArray, j::AbstractSIMD) = fa.val[]
@inline turbo_getindex(fa::FillArray, j::VecUnroll) = fa.val[]
@inline turbo_getindex(fa::FillArray, j) = fa.val[]

@inline function LayoutPointers.memory_reference(fa::FillArray{T}) where T
    p = Ptr{T}(pointer_from_objref(fa.val))
    return p, fa.val
end

@inline function LayoutPointers.stridedpointer_preserve(fa::FillArray{T,N}) where {T,N}
    p, r = LayoutPointers.memory_reference(fa)
    si = _SAI_FA.StrideIndex{N, ntuple(identity, N), 1}(
        ntuple(_ -> StaticInt{0}(), N),   # zero byte-stride
        ntuple(_ -> StaticInt{1}(), N)    # 1-based offsets
    )
    bsi = LayoutPointers.mulstrides(T, si)
    LayoutPointers.stridedpointer(p, bsi, StaticInt{0}()), r
end
