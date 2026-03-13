# using LoopVectorization
using LayoutPointers
using LayoutPointers: StaticInt
import VectorizationBase
const _SAI_FA = LayoutPointers.StaticArrayInterface


# ══════════════════════════════════════════════════════════════════════════════
# UniformArray — mutable single-value array (Ref-backed)
# ══════════════════════════════════════════════════════════════════════════════

"""
    UniformArray{T, N} <: AbstractArray{T,N}

An AbstractArray backed by a single `Ref{T}`. Every index returns the same
value, but unlike `ConstFill`, the value can be changed at runtime via `fa[] = newval`.

Supports `@turbo` via zero-stride `StridedPointer` (same mechanism as `ConstFill`).
"""
struct UniformArray{T, N} <: AbstractArray{T,N}
    val::Base.RefValue{T}
    size::NTuple{N, Int}
end

@inline UniformArray(val::T, size::Int...) where T = UniformArray{T, length(size)}(Ref(val), size)
@inline UniformArray(val::T, size::NTuple{N,Int}) where {T,N} = UniformArray{T, N}(Ref(val), size)
@inline UniformArray(val::T, ::Tuple{}) where T = UniformArray(val)
@inline filltype(::Type{UniformArray}, val, size...; kwargs...) = UniformArray(val, size...)



# ── Base AbstractArray interface ──────────────────────────────────────────────

Base.size(fa::UniformArray) = fa.size
Base.length(fa::UniformArray) = prod(fa.size)

@inline Base.getindex(fa::UniformArray{T,0}) where T = fa.val[]

@inline function Base.getindex(fa::UniformArray{T}, idx::Integer) where T
    @boundscheck checkbounds(fa, idx)
    return fa.val[]
end

@inline function Base.getindex(fa::UniformArray{T}) where T
    return fa.val[]
end

@inline function Base.getindex(fa::UniformArray{T}, idxs::Integer...) where T
    @boundscheck checkbounds(fa, idxs...)
    return fa.val[]
end

@inline function Base.setindex!(fa::UniformArray{T}, v, idx::Integer) where T
    @boundscheck checkbounds(fa, idx)
    fa.val[] = v
end

@inline function Base.setindex!(fa::UniformArray{T}, v) where T
    fa.val[] = v
end

Base.IndexStyle(::Type{<:UniformArray}) = IndexLinear()
Base.similar(fa::UniformArray{T,N}, ::Type{S}, dims::Dims) where {T,N,S} = Array{S}(undef, dims)

Base.strides(::UniformArray{T,N}) where {T,N} = ntuple(_ -> 0, N)
@inline Base.pointer(fa::UniformArray{T}) where T = Ptr{T}(pointer_from_objref(fa.val))

@inline Base.map(f, fa::UniformArray{T}) where T = UniformArray(f(fa.val[]), size(fa))
Base.iterate(fa::UniformArray{T}, state=1) where T = state > length(fa) ? nothing : (fa.val[], state + 1)

# ── LoopVectorization / @turbo integration ────────────────────────────────────

LoopVectorization.check_args(::UniformArray) = true
_SAI_FA.dense_dims(::Type{<:UniformArray{T,N}}) where {T,N} = ntuple(_ -> _SAI_FA.static(true), N)

@inline function LayoutPointers.memory_reference(fa::UniformArray{T}) where T
    p = Ptr{T}(pointer_from_objref(fa.val))
    return p, fa.val
end

@inline function LayoutPointers.stridedpointer_preserve(fa::UniformArray{T,N}) where {T,N}
    p, r = LayoutPointers.memory_reference(fa)
    si = _SAI_FA.StrideIndex{N, ntuple(identity, N), 1}(
        ntuple(_ -> StaticInt{0}(), N),   # zero byte-stride
        ntuple(_ -> StaticInt{1}(), N)    # 1-based offsets
    )
    bsi = LayoutPointers.mulstrides(T, si)
    LayoutPointers.stridedpointer(p, bsi, StaticInt{0}()), r
end
