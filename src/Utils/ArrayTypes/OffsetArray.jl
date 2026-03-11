using LayoutPointers
const _SAI_OA = LayoutPointers.StaticArrayInterface

"""
    OffsetArray{T,N,V} <: AbstractArray{T,N}

Array wrapper with an additive mutable offset offset.
Reads return `offset + vec[i]`, while writes store into `vec` as
`vec[i] = value - offset`.
"""
mutable struct OffsetArray{T,N,V<:AbstractArray{T,N}} <: AbstractArray{T,N}
    vec::V
    offset::T
end

@inline function OffsetArray(vec::V, offset = zero(eltype(vec))) where {V<:AbstractArray}
    T = eltype(vec)
    N = ndims(vec)
    return OffsetArray{T,N,V}(vec, convert(T, offset))
end

@inline filltype(::Type{OffsetArray}, val, size...; offset = 0) = OffsetArray(fill(val, size...), offset)
@inline OffsetArray(offset, vec::V) where {V<:AbstractArray} = OffsetArray(vec, offset)

# ── Base AbstractArray interface ──────────────────────────────────────────────

@inline Base.size(oa::OffsetArray) = size(oa.vec)
@inline Base.length(oa::OffsetArray) = length(oa.vec)
@inline Base.axes(oa::OffsetArray) = axes(oa.vec)
@inline Base.axes(oa::OffsetArray, d::Int) = axes(oa.vec, d)

@inline Base.IndexStyle(::Type{<:OffsetArray{T,N,V}}) where {T,N,V} = Base.IndexStyle(V)

@inline Base.getindex(oa::OffsetArray{T,0}) where {T} = oa.offset + oa.vec[]

@inline function Base.getindex(oa::OffsetArray{T}, idx::Integer) where {T}
    @boundscheck checkbounds(oa, idx)
    return oa.offset + @inbounds oa.vec[idx]
end

@inline function Base.getindex(oa::OffsetArray{T}, idxs::Vararg{Integer,N}) where {T,N}
    @boundscheck checkbounds(oa, idxs...)
    return oa.offset + @inbounds oa.vec[idxs...]
end

@inline function Base.setindex!(oa::OffsetArray{T}, v, idx::Integer) where {T}
    @boundscheck checkbounds(oa, idx)
    @inbounds oa.vec[idx] = convert(T, v) - oa.offset
    return v
end

@inline function Base.setindex!(oa::OffsetArray{T}, v, idxs::Vararg{Integer,N}) where {T,N}
    @boundscheck checkbounds(oa, idxs...)
    @inbounds oa.vec[idxs...] = convert(T, v) - oa.offset
    return v
end

@inline Base.similar(oa::OffsetArray{T,N}, ::Type{S}, dims::Dims) where {T,N,S} = Array{S}(undef, dims)
@inline Base.strides(oa::OffsetArray) = strides(oa.vec)
@inline Base.pointer(oa::OffsetArray) = pointer(oa.vec)

@inline offset(oa::OffsetArray) = oa.offset
@inline function offset!(oa::OffsetArray{T}, v) where {T}
    oa.offset = convert(T, v)
    return oa
end
@inline turbo_components(oa::OffsetArray) = (oa.offset, oa.vec)

@inline function Base.map(f, oa::OffsetArray)
    return OffsetArray(map(f, oa.vec), f(oa.offset))
end

@inline function Base.iterate(oa::OffsetArray, state = 1)
    state > length(oa) && return nothing
    return oa[state], state + 1
end

# ── LoopVectorization / @turbo integration ────────────────────────────────────

LoopVectorization.check_args(::OffsetArray) = false
_SAI_OA.dense_dims(::Type{<:OffsetArray{T,N}}) where {T,N} = ntuple(_ -> _SAI_OA.static(true), N)

"""
Use this in @turbo kernels for OffsetArray reads:
    val = turbo_getindex(v, i)
"""
@inline turbo_getindex(v::AbstractArray, i::Integer) = @inbounds v[i]
@inline turbo_getindex(v::OffsetArray, i::Integer) = @inbounds(v.offset + v.vec[i])
