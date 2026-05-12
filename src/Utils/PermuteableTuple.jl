struct PermuteableTuple{T<:Tuple, PType} <: AbstractVector{Ptype}
    data::T
    permutation::Vector{Int}

    function PermuteableTuple(data::T) where T<:Tuple
        n = length(data)
        ptype = promote_type(T.parameters...)
        return new{T, ptype}(data, collect(1:n))
    end
end

Base.eltype(::Type{PermuteableTuple{T, PType}}) where {T, PType} = PType
Base.getindex(pt::PermuteableTuple, i::Int) = pt.data[pt.permutation[i]]
Base.length(pt::PermuteableTuple) = length(pt.permutation)
Base.iterate(pt::PermuteableTuple, state=1) = state <= length(pt) ? (pt[state], state + 1) : nothing

function Base.permute!(pt::PermuteableTuple, perm)
    permute!(pt.permutation, perm)
end

unshuffled(pt::PermuteableTuple) = pt.data




