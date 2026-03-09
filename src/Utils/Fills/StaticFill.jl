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

    function StaticFill(val::T) where T
        new{val, T, 0}(())
    end

    function StaticFill(val::T, size::Int...) where T
        new{val, T, length(size)}(size)
    end
end

StaticFill(val::T, size::Tuple{}) where T = StaticFill(val)

# -- Base AbstractArray interface ----------------------------------------------

@inline Base.size(sf::StaticFill) = sf.size
@inline Base.length(sf::StaticFill) = prod(sf.size)

@inline Base.getindex(::StaticFill{Val, T, 0}) where {Val, T} = Val::T

@inline Base.map(f, sf::StaticFill{Val, T, N}) where {Val, T, N} = StaticFill(f(Val), size(sf))

@inline Base.@propagate_inbounds function Base.getindex(sf::StaticFill{Val, T, N}, idx::Integer) where {Val, T, N}
    @boundscheck checkbounds(sf, idx)
    return Val::T
end

@inline Base.@propagate_inbounds function Base.getindex(sf::StaticFill{Val, T, N}, idxs::Integer...) where {Val, T, N}
    @boundscheck checkbounds(sf, idxs...)
    return Val::T
end

@inline Base.setindex!(::StaticFill, val, idx...) =
    throw(ArgumentError("Cannot set value of a StaticFill, it is immutable"))

@inline Base.IndexStyle(::Type{<:StaticFill}) = IndexLinear()
@inline Base.similar(sf::StaticFill{V,T,N}, ::Type{S}, dims::Dims) where {V,T,N,S} = Array{S}(undef, dims)

@inline Base.iterate(sf::StaticFill{Val,T}, state=1) where {Val,T} =
    state > length(sf) ? nothing : (Val::T, state + 1)

# -- LoopVectorization / @turbo: direct v[j] path (FastRange) -----------------
#
# FastRange{T}(val, Zero) is special-cased by LV's GroupedStridedPointers:
# vload computes val + 0*(index+offset) which fast-math folds to val.

@inline LoopVectorization.check_args(::StaticFill) = true

@inline function LayoutPointers.stridedpointer_preserve(sf::StaticFill{V,T,N}) where {V,T,N}
    LayoutPointers.FastRange{T}(T(V), StaticInt{0}()), nothing
end
