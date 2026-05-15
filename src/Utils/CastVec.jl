struct CastVec{In,Out} <: AbstractVector{Out}
    data::Vector{In}
end
Base.getindex(v::CastVec{In,Out}, i::Integer) where {In,Out} = Out(v.data[i])
Base.getindex(v::CastVec{In,Out}, range::AbstractRange) where {In,Out} = Out.(v.data[range])
Base.setindex!(v::CastVec{In,Out}, val, i) where {In,Out} = v.data[i] = In(val)
Base.length(v::CastVec) = length(v.data)
Base.size(v::CastVec) = size(v.data)
Base.eltype(::CastVec{In,Out}) where {In,Out} = Out
Base.eltype(::Type{<:CastVec{In,Out}}) where {In,Out} = Out
Base.IteratorSize(::Type{<:CastVec}) = Base.HasLength()
Base.iterate(v::CastVec, i=1) = i > length(v) ? nothing : (v[i], i+1)
CastVec(t::Type, data) = CastVec{eltype(data), t}(data)
Base.convert(::Type{CastVec{In,Out}}, data::Vector{In}) where {In,Out} = CastVec{In,Out}(data)
Base.convert(::Type{CastVec{In,Out}}, data::AbstractVector) where {In,Out} =
    CastVec{In,Out}(Vector{In}(data))
