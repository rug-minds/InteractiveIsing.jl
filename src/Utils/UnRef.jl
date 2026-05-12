mutable struct UnRef{T}
    val::Union{Nothing, T}
    destructor::Function
end

UnRef(val::Any) = UnRef{typeof(val)}(val, (v) -> nothing)
UnRef(tp::DataType) = UnRef{tp}(nothing, (v) -> nothing)
UnRef(tp::DataType, destructor::Function) = UnRef{tp}(nothing, destructor)

reset!(u::UnRef) = begin u.destructor(u.val); u.val = nothing; end
reset!(u::UnRef{T}, val::T) where T = begin u.destructor(u.val); u.val = val; end

Base.isnothing(u::UnRef) = isnothing(u.val)

Base.getindex(u::UnRef{T}) where T = u.val::T
function Base.setindex!(u::UnRef{T}, val::T) where T
    if isnothing(u.val)
        u.val = val
    else
        error("UnRef already has a value")
    end
end

