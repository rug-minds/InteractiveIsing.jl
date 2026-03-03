using LoopVectorization
using LayoutPointers
using LayoutPointers: StaticInt
import VectorizationBase
using VectorizationBase: AbstractSIMD, VecUnroll, vload
const _SAI = LayoutPointers.StaticArrayInterface


"""
    StaticFill{Val, T, N} <: AbstractArray{T,N}

An AbstractArray whose value is baked into the type parameter `Val`.
Every index returns the same compile-time constant.  Useful as a drop-in
replacement for `Vector`/`Array` when a parameter is spatially uniform,
enabling the compiler to constant-fold and SIMD-broadcast the value.

## Usage with `@turbo`

Two integration paths:

1. **Direct `v[j]` syntax** — uses a zero-step FastRange via
   `stridedpointer_preserve`.  Works well for dense contiguous loops.
   For sparse/indirect indexing LLVM may not DCE the (unused) index load,
   leaving residual overhead vs. a literal constant.

2. **`turbo_getindex(v, j)`** — a compute-op alternative that @turbo calls
   as an opaque function.  For StaticFill the index arg is ignored and a scalar
   constant is returned; after inlining, LLVM can DCE the entire index chain.
   Use this for literal-speed sparse/indirect @turbo loops.
"""
struct StaticFill{Val, T, N} <: AbstractArray{T,N}
    size::NTuple{N, Int}

    function StaticFill(val::T, size::Int...) where T
        new{val, T, length(size)}(size)
    end
end

# -- Base AbstractArray interface ----------------------------------------------

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

@inline Base.setindex!(::StaticFill, val, idx...) =
    throw(ArgumentError("Cannot set value of a StaticFill, it is immutable"))

Base.IndexStyle(::Type{<:StaticFill}) = IndexLinear()
Base.similar(sf::StaticFill{V,T,N}, ::Type{S}, dims::Dims) where {V,T,N,S} = Array{S}(undef, dims)

Base.iterate(sf::StaticFill{Val,T}, state=1) where {Val,T} =
    state > length(sf) ? nothing : (Val::T, state + 1)

# -- LoopVectorization / @turbo: direct v[j] path (FastRange) -----------------
#
# FastRange{T}(val, Zero) is special-cased by LV's GroupedStridedPointers:
# vload computes val + 0*(index+offset) which fast-math folds to val.

LoopVectorization.check_args(::StaticFill) = true

@inline function LayoutPointers.stridedpointer_preserve(sf::StaticFill{V,T,N}) where {V,T,N}
    LayoutPointers.FastRange{T}(T(V), StaticInt{0}()), nothing
end

# -- turbo_getindex: compute-op path for @turbo -----------------------------
#
# Use  turbo_getindex(v, j)  instead of  v[j]  inside @turbo loops.
#
# @turbo treats this as an opaque function call (compute-op):
#   * v  is hoisted as a loop constant
#   * j  arrives as a SIMD type (Vec, VecUnroll, MM, ...)
#   * the function is called directly with those types
#
# For StaticFill the index is ignored and a scalar constant is returned.
# After inlining, LLVM sees that the index-load result is dead and can DCE
# the entire gather from rowval (or whatever produced the index).
#
# For AbstractVector the call forwards to VectorizationBase.vload, giving
# the same gather / sequential load that @turbo would emit natively.

# -- StaticFill path: return scalar constant, ignore index entirely --
# These must come BEFORE AbstractVector methods to resolve ambiguity
# (StaticFill <: AbstractVector for N=1).
@inline turbo_getindex(sf::StaticFill{V,T}, j::Integer) where {V,T} = T(V)
@inline turbo_getindex(sf::StaticFill{V,T}, j::AbstractSIMD) where {V,T} = T(V)
@inline turbo_getindex(sf::StaticFill{V,T}, j::VecUnroll) where {V,T} = T(V)
@inline turbo_getindex(sf::StaticFill{V,T}, j) where {V,T} = T(V)

# -- AbstractVector paths (scalar + SIMD) --
@inline turbo_getindex(v::AbstractVector, j::Integer) = @inbounds v[j]

@inline function turbo_getindex(v::AbstractVector, j::AbstractSIMD)
    vload(LayoutPointers.stridedpointer(v), (j,))
end

@inline function turbo_getindex(v::AbstractVector, j::VecUnroll)
    vload(LayoutPointers.stridedpointer(v), (j,))
end

# Tell @turbo that turbo_getindex is safe for SIMD loops.
# can_turbo's default check (promote_op with Vec args) fails because
# the first argument is an array, not a scalar.
LoopVectorization.can_turbo(::typeof(turbo_getindex), ::Val{2}) = true
