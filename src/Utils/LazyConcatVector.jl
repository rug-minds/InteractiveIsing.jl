"""
    LazyConcatVector(first, second)

Read-only `AbstractVector` view of two vectors as if they were concatenated.
No storage is copied: indices `1:length(first)` read from `first`, and the
remaining indices read from `second`.
"""
struct LazyConcatVector{T,A<:AbstractVector,B<:AbstractVector} <: AbstractVector{T}
    first::A
    second::B
end

function LazyConcatVector(first::A, second::B) where {A<:AbstractVector,B<:AbstractVector}
    T = promote_type(eltype(A), eltype(B))
    return LazyConcatVector{T,A,B}(first, second)
end

Base.IndexStyle(::Type{<:LazyConcatVector}) = IndexLinear()
Base.size(v::LazyConcatVector) = (length(v.first) + length(v.second),)
Base.length(v::LazyConcatVector) = length(v.first) + length(v.second)

@inline function Base.getindex(v::LazyConcatVector, i::Int)
    @boundscheck checkbounds(v, i)
    nfirst = length(v.first)
    return i <= nfirst ? v.first[i] : v.second[i - nfirst]
end

function Base.iterate(v::LazyConcatVector, state::Int = 1)
    state > length(v) && return nothing
    return (v[state], state + 1)
end

