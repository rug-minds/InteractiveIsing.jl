mutable struct UnRef{T}
    val::Union{Nothing, T}
end

UnRef(tp::DataType) = UnRef{tp}(nothing)

reset!(u::UnRef) = u.val = nothing

Base.isnothing(u::UnRef) = isnothing(u.val)

Base.getindex(u::UnRef{T}) where T = u.val::T
function Base.setindex!(u::UnRef{T}, val::T) where T
    if isnothing(u.val)
        u.val = val
    else
        error("UnRef already has a value")
    end
end

